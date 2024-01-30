from rrt import RRT
from rrt_star import RRTStar
from trrt_star import TRRTStar

def get_lp_planner(start=None, goal=None, bounds=None, type=None, 
                   max_extend_length=5.0,
                   goal_p_th=0.5,
                   goal_sample_rate=0.1,    
                   max_iter=100,                 
                   path_tick=0.1,                    
                   polygons=[],
                   bias_sampling=False, cost_th=0.9,
                   height_fn=None, cost_fn=None, constraint_fn=None, 
                   animation=False, try_goal=False, t_weight=10.0,
                #    curvature=1.0, robot_radius=1.5, robot_sensor_range=20.0
                   ):

    if type == 'RRT':
        planner = RRT(start, goal, bounds,                       
                      max_extend_length=max_extend_length, 
                      goal_p_th=goal_p_th, 
                      goal_sample_rate=goal_sample_rate, 
                      max_iter=max_iter,
                      path_tick=path_tick, 
                      animation=animation,
                      height_fn=height_fn,                       
                      )
    elif type == 'RRTSTAR':
        planner = RRTStar(start, goal, bounds, 
                          max_extend_length=max_extend_length, 
                          goal_p_th=goal_p_th, 
                          goal_sample_rate=goal_sample_rate, 
                          max_iter=max_iter,
                          path_tick=path_tick, 
                          animation=animation, 
                          height_fn=height_fn, 
                          try_goal=try_goal, 
                          connect_circle_dist=10,
                          t_weight=t_weight,                           
                          )
    elif type == 'TRRTSTAR':
        planner = TRRTStar(start, goal, bounds, 
                           max_extend_length=max_extend_length,
                           goal_p_th=goal_p_th,
                           goal_sample_rate=goal_sample_rate, 
                           max_iter=max_iter,
                           path_tick=path_tick,                            
                           animation=animation, 
                           height_fn=height_fn, 
                           try_goal=try_goal,
                           connect_circle_dist=100,                           
                           cost_fn=cost_fn, 
                           cost_th = cost_th,                           
                           constraint_fn=constraint_fn,                           
                           bias_sampling=bias_sampling, 
                           t_weight=t_weight,
                           polygons=polygons,                           
                           )
    else:
        print(f"Wrong type: {type}")
    return planner