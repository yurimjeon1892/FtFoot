import math

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from path_manager import compute_interp_path_from_wp

class RRT:
    class Node:
        def __init__(self, p):
            self.p = np.array(p)
            self.parent = None
            self.path = []

        def __repr__(self) -> str:
            return f'pos: {self.p}, path: {self.path}'

    def __init__(self, start, goal, bounds, 
                 max_extend_length=5.0,
                 goal_p_th=0.1,
                 goal_sample_rate=0.1,
                 max_iter=100,
                 path_tick=0.1,
                 animation=False,
                 height_fn=None,
                 dem_fn=None):
        self.bounds = bounds        
        self.max_extend_length = max_extend_length
        self.goal_p_th = goal_p_th
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []
        self.path_tick = path_tick
        self.animation = animation
        self.height_fn = height_fn        
        self.dem_fn = dem_fn            

        self.set_start_goal(start, goal)

    def set_start_goal(self, start, goal):
        if start is not None and goal is not None:
            self.start = self.Node(start[:2])
            self.goal = self.Node(goal[:2])

    def plan(self):
        """Plans the path from start to goal while avoiding obstacles"""
        self.goal_iter = -1
        self.node_list = [self.start]
        with tqdm(total=self.max_iter, desc='planning local path...') as pbar:
            for i in range(self.max_iter):
                # 1) Create a random node (rnd_node) inside
                # the bounded environment
                pbar.update(1)
                rnd_node = self.get_random_node()

                # 2) Find nearest node (nearest_node)
                nearest_node = self.get_nearest_node(self.node_list, rnd_node)

                # 3) Get new node (new_node) by connecting
                # rnd_node and nearest_node. Hint: steer
                new_node = self.steer(nearest_node, rnd_node, self.max_extend_length)
                # 4) If the path between new_node and the
                # nearest node is not in collision, add it to the node_list
                if new_node is not None and not self.collision(new_node, nearest_node):
                    self.node_list.append(new_node)
                    if self.animation:
                        self.draw_graph(new_node)
                # Don't need to modify beyond here
                # If the new_node is very close to the goal, connect it
                # directly to the goal and return the final path
                if self.dist_to_goal(self.node_list[-1].p) <= self.goal_p_th:
                    if not self.collision(self.goal, self.node_list[-1]):
                        final_path = self.final_path(len(self.node_list) - 1)
                        self.goal_iter = i

                        if self.animation:
                            self.plot_path(final_path)
                            # plt.show()
                            plt.close()
                        return final_path, self.path_length(final_path)
                    
        # last_index, min_cost = self.best_goal_node_index()
        # return final_path, self.path_length(final_path)
        return None  # cannot find path

    def steer(self, from_node, to_node, max_extend_length=np.inf):
        """Connects from_node to a new_node in the direction of to_node
        with maximum distance max_extend_length
        """
        new_node = self.Node(to_node.p)
        d = from_node.p - to_node.p
        dist = np.linalg.norm(d)
        if dist > max_extend_length:
            # rescale the path to the maximum extend_length
            new_node.p  = from_node.p - d / dist * max_extend_length
            dist = max_extend_length

        path = compute_interp_path_from_wp(start_xp=[from_node.p[0], new_node.p[0]],
                                           start_yp=[from_node.p[1], new_node.p[1]], step=self.path_tick)
        if len(path) > 0:
            path = np.insert(path, 0, [from_node.p[0], from_node.p[1], path[0,-1]], axis=0)
        else:
            return None

        new_node.parent = from_node
        new_node.path = path

        return new_node

    def dist_to_goal(self, p):
        """Distance from p to goal"""
        return np.linalg.norm(p - self.goal.p)

    def get_random_node(self):
        """Sample random node inside bounds or sample goal point"""
        if np.random.rand() > self.goal_sample_rate:
            # Sample random point inside boundaries
            upper = np.array([self.bounds[1], self.bounds[3]])
            lower = np.array([self.bounds[0], self.bounds[2]])
            sample = np.random.rand(2)*(upper-lower) + lower

            z_sample = self.height_fn(sample)[2]
            if z_sample == -np.inf:
                return self.get_random_node()

            rnd = self.Node(sample)
        else:
            # Select goal point
            rnd = self.Node(self.goal.p)
        return rnd

    @staticmethod
    def get_nearest_node(node_list, node):
        """Find the nearest node in node_list to node"""
        dlist = [np.sum(np.square((node.p - n.p))) for n in node_list]
        minind = dlist.index(min(dlist))
        return node_list[minind]

    def collision(self, node1, node2):
        """Check whether the path connecting node1 and node2 is in collision"""
        p1 = node2.p
        p2 = node1.p

        p1 = self.height_fn(p1)
        p2 = self.height_fn(p2)

        dist = np.linalg.norm(p1 - p2)

        if np.isnan(dist) :
            return True

        if self.dem_fn is not None:
            edge_node = self.steer(node2, node1)
            edge_points = edge_node.path[:,:2] if edge_node else np.array([p1, p2])
            for point in edge_points:
                if np.isnan(self.dem_fn(point)):
                    return True

        return False # is not in collision

    def final_path(self, goal_ind):
        """Compute the final path from the goal node to the start node"""
        # path = [self.goal.p]
        path = []
        node = self.node_list[goal_ind]
        # modify here: Generate the final path from the goal node to the start node.
        # We will check that path[0] == goal and path[-1] == start
        while node.parent is not None:
        #   path.append(node.p)
            for r_path in reversed(node.path):
                pz = self.height_fn(r_path[:2])[2] if self.height_fn else 0
                path.append([r_path[0],r_path[1],pz,r_path[2],1]) # x,y,z,yaw,gear
            node = node.parent
        # path.append(self.start.p)
        path.reverse()
        return np.array(path)

    def draw_graph(self, rnd=None, z=False):
        if z:
            for node in self.node_list:
                if node.parent and len(node.path) != 0 and self.height_fn:
                    pz = np.array([self.height_fn(ip)[2] for ip  in node.path[:, :2]])
                    mask = pz != -np.inf
                    plt.plot(node.path[mask,0], node.path[mask,1], pz[mask], "-g", alpha=0.5)
        else:
            plt.clf()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                        lambda event: [exit(0) if event.key == 'escape' else None])
            if rnd is not None:
                plt.plot(rnd.p[0], rnd.p[1], "^k")
            for node in self.node_list:
                if node.parent and len(node.path) != 0:
                    plt.plot(node.path[:,0], node.path[:,1], "-g")            

            plt.plot(self.start.p[0], self.start.p[1], "xr")
            plt.plot(self.goal.p[0], self.goal.p[1], "xb")
            plt.axis("equal")
            plt.grid(True)
            if self.animation:
                plt.pause(1e-6)

    @staticmethod
    def plot_scene(start, goal, bounds, z=False):
        if z:
            ax = plt.gca(projection='3d', adjustable='box')
            # ax.set_aspect('equal')
            ax.plot(start[0], start[1], start[2], "*r", markersize=15)
            ax.plot(goal[0], goal[1], goal[2], "*b", markersize=15)
            plt.legend(('start', 'goal'), loc='upper left')
            ax.set_xlim3d([bounds[0], bounds[1]])
            ax.set_ylim3d([bounds[2], bounds[3]])
            ax.set_zlim3d([bounds[4], bounds[5]])
            ax.set_box_aspect([bounds[1] - bounds[0],bounds[3] - bounds[2], 10])
        else:                        
            plt.axis([bounds[0]-0.5, bounds[1]+0.5, bounds[2]-0.5, bounds[3]+0.5])
            plt.plot(start[0], start[1], "*r", markersize=15)
            plt.plot(goal[0], goal[1], "*b", markersize=15)
            plt.legend(('start', 'goal'), loc='upper left')
            plt.gca().set_aspect('equal')

    @staticmethod
    def plot_path(path, verbose=False, z=False):
        if verbose:
            print(f'optimal path : {path}')
        if z:
            plt.plot(path[:,0], path[:,1], path[:,2], '-r')
        else:
            plt.plot(path[:,0], path[:,1], '-r')

    @staticmethod
    def path_length(path):
        return np.linalg.norm(np.diff(path, axis=0)[:, :2], axis=-1).sum()

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta