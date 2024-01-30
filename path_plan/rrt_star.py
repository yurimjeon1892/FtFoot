from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from rrt import RRT

class RRTStar(RRT):
    class Node(RRT.Node):
        def __init__(self, p):
            super().__init__(p)
            self.cost = 0.0

        def __repr__(self) -> str:
            return f'{self.p}'
    def __init__(self, start, goal, bounds, 
                 max_extend_length=5.0,
                 goal_p_th=5,
                 goal_sample_rate=0.1,
                 max_iter=100,
                 path_tick=0.1,
                 animation=False,
                 height_fn=None,
                 try_goal=False,
                 connect_circle_dist=50.0,
                 t_weight=10.0):
        super().__init__(start, goal, bounds, max_extend_length,
                         goal_p_th, goal_sample_rate,max_iter, path_tick, animation, height_fn)
        self.try_goal = try_goal
        self.connect_circle_dist = connect_circle_dist        
        self.t_weight = t_weight

    def plan(self):        
        last_index, min_cost = None, None
        goal_flag = False
        self.goal_iter = -1
        self.node_list = [self.start]
        with tqdm(total=self.max_iter, desc='planning local path...') as pbar:
            for i in range(self.max_iter):
                # Create a random node inside the bounded environment                
                pbar.update(1)
                rnd = self.get_random_node()
                # Find nearest node
                nearest_node = self.get_nearest_node(self.node_list, rnd)
                # Get new node by connecting rnd_node and nearest_node
                new_node = self.steer(nearest_node, rnd, self.max_extend_length)
                # If path between new_node and nearest node is not in collision:
                if new_node is not None and not self.collision(new_node, nearest_node):
                    near_inds = self.near_nodes_inds(new_node)
                    # Connect the new node to the best parent in near_inds
                    new_node = self.choose_parent(new_node, near_inds)
                    if new_node is None:
                        continue
                    self.node_list.append(new_node)
                    # Rewire the nodes in the proximity of new_node if it improves their costs
                    self.rewire(new_node, near_inds)
                    if self.try_goal: #optional
                        self.try_goal_path(new_node)

                    if not goal_flag:
                        last_index, min_cost = self.best_goal_node_index()
                        if last_index is not None:
                            goal_flag = True
                            self.goal_iter = i

                    if self.animation:
                        self.draw_graph(new_node)

                if goal_flag:
                    pbar.set_description(desc=f"\033[96mGoal founded at {self.goal_iter} iteration\033[00m ")

        last_index, min_cost = self.best_goal_node_index()
        if last_index:
            final_path = self.final_path(last_index)
            if self.animation:
                self.plot_path(final_path)
                # plt.show()
                plt.close()
            return final_path, min_cost
        return None, min_cost

    def choose_parent(self, new_node, near_inds):
        """Set new_node.parent to the lowest resulting cost parent in near_inds and
        new_node.cost to the corresponding minimal cost
        """
        if not near_inds:
            return None

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node, self.max_extend_length)
            if t_node and not self.collision(t_node, near_node):
                costs.append(self.new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node, self.max_extend_length)
        new_node.cost = min_cost

        return new_node

    def rewire(self, new_node, near_inds):
        """Rewire near nodes to new_node if this will result in a lower cost"""
        # modify here: Go through all near nodes and check whether rewiring them
        # to the new_node would:
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node, self.max_extend_length)
            if not edge_node:
                continue
            # A) Not cause a collision and
            not_collision = not self.collision(new_node, near_node)

            # B) reduce their own cost.
            edge_node.cost = self.new_cost(new_node, near_node)
            improved_cost = near_node.cost > edge_node.cost

            # If A and B are true, update the cost and parent properties of the node.
            if not_collision and improved_cost:
                near_node.p = edge_node.p
                near_node.cost = edge_node.cost
                near_node.path = edge_node.path
                near_node.parent = edge_node.parent
                # Don't need to modify beyond here
                self.propagate_cost_to_leaves(new_node)

    def best_goal_node_index(self):
        """Find the lowest cost node to the goal"""
        min_cost = np.inf
        best_goal_node_idx = None
        for i in range(len(self.node_list)):
            node = self.node_list[i]
            # Has to be in close proximity to the goal
            if self.dist_to_goal(node.p) <= self.goal_p_th:
                # Connection between node and goal needs to be collision free
                if not self.collision(self.goal, node):
                    # The final path length
                    cost = node.cost + self.dist_to_goal(node.p)
                    if node.cost + self.dist_to_goal(node.p) < min_cost:
                        # Found better goal node!
                        min_cost = cost
                        best_goal_node_idx = i
        return best_goal_node_idx, min_cost

    def near_nodes_inds(self, new_node):
        """Find the nodes in close proximity to new_node"""
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * np.sqrt((np.log(nnode) / nnode))
        dlist = [np.sum(np.square((node.p - new_node.p))) for node in self.node_list]
        near_inds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return near_inds

    def new_cost(self, from_node, to_node):
        """to_node's new cost if from_node were the parent"""
        d = np.linalg.norm(from_node.p - to_node.p)
        if self.dem_fn is not None:
            d = self.t_distance(from_node, to_node, d)
        return from_node.cost + d

    def t_distance(self, from_node, to_node, distance, eps=1e-6, thres=1.65):
        edge_node = self.steer(from_node, to_node)
        edge_points = edge_node.path[:,:2] if edge_node else np.array([from_node.p, to_node.p])
        t_score = np.array([self.dem_fn(point) for point in edge_points])

        zt_score = (t_score - np.mean(t_score)) / max(np.std(t_score), eps)
        zt_mask = np.logical_and(zt_score < thres, zt_score > -thres)

        t_score = np.max(t_score[zt_mask])
        t_score = 1 / max(1 - t_score, eps) - 1
        t_dist = (1 + self.t_weight * t_score) * distance
        return t_dist

    def propagate_cost_to_leaves(self, parent_node):
        """Recursively update the cost of the nodes"""
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def try_goal_path(self, node):
        new_node = self.steer(node, self.goal)
        if new_node is None:
            return
        if not self.collision(new_node, node):
            self.node_list.append(new_node)