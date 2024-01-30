# import os
# import sys
# from threading import local

import numpy as np
# from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Poly
# import pyransac3d as pyrsc
# from scipy.stats import wasserstein_distance

from rrt_star import RRTStar
from misc import in_poly, process, normalization
from path_manager import compute_interp_path_from_wp
# from models.path_tracker.mpc import MPC

class TRRTStar(RRTStar):
    def __init__(self, start, goal, bounds, 
                 max_extend_length=5.0,
                 goal_p_th=0.5,
                 goal_sample_rate=0.1,
                 max_iter=100,
                 path_tick=0.1,
                 animation=False,
                 height_fn=None,
                 try_goal=False,
                 connect_circle_dist=50.0,
                 cost_fn=None,
                 cost_th=0.7,
                 constraint_fn=None,
                 bias_sampling=False,
                 t_weight=1.,
                 polygons=[]) -> None:
        super().__init__(start, goal, bounds, max_extend_length,
                         goal_p_th, goal_sample_rate,max_iter, path_tick, animation,
                         height_fn, try_goal, connect_circle_dist)
        self.cost_fn = cost_fn
        self.constraint_fn = constraint_fn
        self.bias_sampling = bias_sampling
        self.t_weight = t_weight
        self.polygons = polygons
        self.cost_th = cost_th

    def set_start_goal(self, start, goal):
        if start is not None and goal is not None:
            self.start = self.Node(start)
            self.goal = self.Node(goal)

    def get_random_node(self):
        """Sample random node inside bounds or sample goal point"""
        if np.random.rand() > self.goal_sample_rate:
            # Sample random point inside boundaries
            if self.bias_sampling:
                sample = self.rejection_sampling()
            else:
                sample = self.uniform_sampling()

            sample = self.height_fn(sample)
            if sample[2] == -np.inf or self.cost_fn(sample) >= self.cost_th :
                return self.get_random_node()
            rnd = self.Node(sample)
        else:
            # Select goal point
            rnd = self.Node(self.goal.p)
        return rnd

    def rejection_sampling(self):
        upper = np.array([self.bounds[1], self.bounds[3]])
        lower = np.array([self.bounds[0], self.bounds[2]])
        q_sample = np.random.rand(2)*(upper-lower) + lower

        if self.constraint_fn is not None and not self.constraint_fn(q_sample):
            return self.rejection_sampling()

        p = 1 - self.cost_fn(q_sample)
        u = np.random.rand()
        if u < p:
            return q_sample
        else:
            return self.rejection_sampling()

    def uniform_sampling(self):
        upper = np.array([self.bounds[1], self.bounds[3]])
        lower = np.array([self.bounds[0], self.bounds[2]])
        sample = np.random.rand(2)*(upper-lower) + lower
        if self.constraint_fn is not None and not self.constraint_fn(sample):
            return self.uniform_sampling()
        return sample

    def new_cost(self, from_node, to_node):
        """to_node's new cost if from_node were the parent"""
        distance = np.linalg.norm(to_node.p - from_node.p)
        d = self.t_distance(from_node, to_node, distance)
        return from_node.cost + d

    def t_distance(self, from_node, to_node, distance):
        edge_node = self.steer(from_node, to_node, self.max_extend_length)
        edge_points = edge_node.path if edge_node else np.array([from_node.p, to_node.p])

        t_score = np.array([self.cost_fn(point) for point in edge_points])

        # zt_score = (t_score - np.mean(t_score)) / max(np.std(t_score), eps)
        # zt_mask = np.logical_and(zt_score < thres, zt_score > -thres)

        t_score = np.max(t_score)
        # print(t_score, distance)
        # t_score = 1 / max(1 - t_score, eps) - 1
        # t_dist = (1 + self.t_weight * t_score) * distance
        t_dist = self.t_weight * t_score + distance
        return t_dist

    def collision(self, to_node, from_node):
        """Check whether the path connecting node1 and node2 is in collision"""
        # if len(self.polygons) != 0:
        p1, p2 = from_node.p[:2], to_node.p[:2]            
        edge_node = self.steer(from_node, to_node, self.max_extend_length)
        edge_points = edge_node.path[:,:3] if edge_node else np.array([p1, p2])
        for point in edge_points:
            # if self.cost_fn(point) == 1.0:
            if self.cost_fn(point) > self.cost_th:
                return True
            if self.check_polygons(point[:2]):
                return True

        return False # is not in collision

    def steer(self, from_node, to_node, max_extend_length=np.inf):
        """Connects from_node to a new_node in the direction of to_node
        with maximum distance max_extend_length
        """
        new_node = self.Node(to_node.p)
        d = from_node.p - to_node.p
        dist = np.linalg.norm(d)
        if dist > max_extend_length:
            # rescale the path to the maximum extend_length
            tmp_node = from_node.p - d / dist * max_extend_length
            new_node.p = self.height_fn(tmp_node[:2])

        path = compute_interp_path_from_wp(start_xp=[from_node.p[0], new_node.p[0]],
                                           start_yp=[from_node.p[1], new_node.p[1]], step=self.path_tick)

        if len(path) > 0:
            path = np.insert(path, 0, [from_node.p[0], from_node.p[1], path[0,-1]], axis=0) # x,y,yaw

            path = np.stack([np.concatenate([self.height_fn(pt[:2]), [pt[2]]]) for pt in path]) # x,y,z,yaw
        else:
            return None

        new_node.parent = from_node
        new_node.path = path # swap column : x, y, z, yaw
        
        d = from_node.p - new_node.p
        dist = np.linalg.norm(d)
        return new_node

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
                path.append([*r_path,1]) # x,y,z,yaw,gear
            node = node.parent
        # path.append(self.start.p)
        path.reverse()
        return np.array(path)

    def update_polygons(self, polygons):
        self.polygons = polygons

    def check_polygons(self, node):
        if len(self.polygons) != 0:
            return np.any([in_poly(poly, node) for poly in self.polygons])
        else:
            return False

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

            if self.bounds is not None:
                if self.animation:
                    plt.plot([self.bounds[0], self.bounds[1], self.bounds[1], self.bounds[0], self.bounds[0]],
                            [self.bounds[2], self.bounds[2], self.bounds[3], self.bounds[3], self.bounds[2]],
                            "-k")

            plt.plot(self.start.p[0], self.start.p[1], "xr")
            plt.plot(self.goal.p[0], self.goal.p[1], "xb")

            ax = plt.gca()
            for poly in self.polygons:
                p = Poly(xy=poly, facecolor = 'k')
                ax.add_patch(p)

            plt.axis("equal")
            plt.grid(True)
            if self.animation:
                plt.pause(0.01)
