import os, inspect
from dataclasses import dataclass

import yaml
import random
import numpy as np
from matplotlib.path import Path

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

@dataclass
class Joint:
    index: int
    name: str
    type: int
    gIndex: int
    uIndex: int
    flags: int
    damping: float
    friction: float
    lowerLimit: float
    upperLimit: float
    maxForce: float
    maxVelocity: float
    linkName: str
    axis: tuple
    parentFramePosition: tuple
    parentFrameOrientation: tuple
    parentIndex: int

    def __post_init__(self):
        self.name = str(self.name, 'utf-8')
        self.linkName = str(self.linkName, 'utf-8')

def get_local_map(state, global_height_map, detection_range, args):
    """
    Computes the local height map around vehicle
    """
    margin = int(detection_range / args['resolution'])
    cx = int((state.x - args['x0']) / args['resolution'])
    cy = int((state.y - args['y0']) / args['resolution'])

    min_x, min_y = max(cx - margin, 0)                , max(cy - margin, 0)
    max_x, max_y = min(cx + margin, args['map_width']), min(cy + margin, args['map_height'])
    local_height_map = global_height_map[min_x : max_x, min_y : max_y]

    return local_height_map

def get_directory():
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent_dir = os.path.dirname(current_dir)
    output_dir = os.path.join(parent_dir, 'outputs')
    env_dir = os.path.join(parent_dir, 'envs')
    os.makedirs(output_dir, exist_ok=True)
    return current_dir, parent_dir, output_dir, env_dir

def getInitOrientationFromVector(u1, u3 = [0,0,1]):
    u1 = u1 / np.linalg.norm(u1)
    u3 = np.array(u3)
    u2 = np.cross(u3, u1)
    trans_mat = np.array([u1,u2,u3]).T
    initial_orient = getQuaternionFromRot(trans_mat)
    return initial_orient

def getQuaternionFromVectors(u, v):
    norm_u_norm_v = np.sqrt(np.dot(u,u) * np.dot(v,v)) + 1e-8
    cos_theta = np.dot(u,v) / norm_u_norm_v
    halfcos = np.sqrt(0.5 * (1 + cos_theta)) + 1e-8
    w = np.cross(u,v) * (1 / (norm_u_norm_v * 2 * halfcos))
    return [*w, halfcos]

def getQuaternionFromRot(rot):
    "http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/"
    [m00, m01, m02],[m10, m11, m12], [m20, m21, m22] = rot
    tr = m00 + m11 + m22
    if (tr > 0):
        S = np.sqrt(tr + 1.0) * 2 # S=4*qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif ((m00 > m11) and (m00 > m22)):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2 # S=4*qx
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif (m11 > m22):
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2 # S=4*qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2 # S=4*qz
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return [qx,qy,qz,qw]

def RotationwithQuaternion(q, p):
    x,y,z,w = q
    R_q = np.array([[w,z,-y,x],[-z,w,x,y],[y,-x,w,z],[-x,-y,-z,w]])
    L_q = np.array([[w,z,-y,-x],[-z,w,x,-y],[y,-x,w,-z],[x,y,z,w]])
    return p @ R_q @ L_q

def MultiplyQuaternion(q1,q2):
    a1,b1,c1,d1 = q1
    a2,b2,c2,d2 = q2
    x1 = a1*a2 - b1*b2 - c1*c2 - d1*d2
    x2 = a1*b2 + b1*a2 + c1*d2 - d1*c2
    x3 = a1*c2 - b1*d2 + c1*a2 + d1*b2
    x4 = a1*d2 + b1*c2 - c1 *b2 + d1*a2
    return np.array([x1,x2,x3,x4])

def set_local_frame(p, robot, lineWidth=2.0, lifeTime=0.1):
#     start, orient = p.getBasePositionAndOrientation(robot)
    start, orient, *_ = p.getLinkState(robot, 1) # 1 is base frame of imu
    rot_mat = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)
    colors = np.eye(3)
    ends = start + rot_mat.T@colors
    for i, end in enumerate(ends):
        p.addUserDebugLine(start,
                           end,
                           colors[i],
                           parentObjectUniqueId=robot,
                           lineWidth=lineWidth,
                           lifeTime=lifeTime)

def set_local_camera(p, robot, distance=4, yaw_offset=0, pitch_offset=-35):
    pos, orient = p.getBasePositionAndOrientation(robot)
    yaw = p.getEulerFromQuaternion(orient)[2]
    yaw = -90 + np.rad2deg(yaw) + yaw_offset
    p.resetDebugVisualizerCamera(distance,
                                 cameraYaw=yaw,
                                 cameraPitch=pitch_offset,
                                 cameraTargetPosition=pos)

def load_env_yaml(envdir, args):
    angle = get_angles(args)
    env_args = {}
    post_fix = '_s' + str(args['scale']) \
                + '_r' + str(angle['rolling_degree']) \
                + '_p' + str(angle['pitching_degree']) \
                + '_y' + str(angle['yawing_degree']) \
                + '_f' + str(args['filter_size'])
    with open(os.path.join(envdir, 'humji', args['terrain_name'], args['terrain_name'] + post_fix + '.yaml')) as f:
        env_args.update(yaml.safe_load(f))
    return env_args

def get_angles(args):
    elevation = args['elevation']
    angle_info = args['change_angle'][args['terrain_name']]
    angle = {}
    change_angle = list(angle_info.keys())[0]
    angle.update(angle_info[change_angle]['default'])
    angle.update({change_angle: angle_info[change_angle][elevation]})
    return angle

def in_poly(vertice, points):
    # counter clock wise path needed : https://github.com/matplotlib/matplotlib/issues/9704
    path = Path(vertice, closed=True)
    if len(points.shape) > 1:
        inside = path.contains_points(points, radius=1e-9)
    else:
        inside = path.contains_points([points], radius=1e-9)
    return inside

def normalization(x):
    minimum = np.nanmin(x)
    maximum = np.nanmax(x)
    return (x - minimum) / (maximum - minimum)

def standardization(x):
    std = np.nanstd(x)
    mean = np.nanmean(x)
    mask = ~np.isnan(x)
    x[mask] = (x[mask] - mean) / std
    return x

def process(x, cliped_value=2):
    x = standardization(x)
    x = np.clip(x, a_min=-cliped_value, a_max=cliped_value)
    x = normalization(x)
    return x

def mean_pooling(arr, stride):
    w,h = arr.shape
    new_w, new_h = w // stride, h // stride
    arr = np.nanmean(arr.reshape(new_w, stride, new_h, stride), axis=(1,3))
    return arr

def max_pooling(arr, stride):
    w,h = arr.shape
    new_w, new_h = w // stride, h // stride
    arr = np.nanmax(arr.reshape(new_w, stride, new_h, stride), axis=(1,3))
    return arr

def filter_pooling(arr, stride, s_min):
    arr = mean_pooling(arr, stride)
    thr = s_min / (stride * stride)
    arr = arr >= thr
    return arr

def unpooling(arr, stride):
    arr = np.repeat(arr, stride, axis=1).repeat(stride, axis=0)
    return arr

def sifting(arr, sift, pad=False):
    w = arr.shape[0]
    h = arr.shape[1]
    new_w = w // sift
    new_h = h // sift
    x = np.linspace(0, w-1, new_w).astype(int)
    y = np.linspace(0, h-1, new_h).astype(int)
    XX, YY = map(np.transpose, np.meshgrid(x,y))
    new_arr = np.full_like(arr, pad)
    new_arr[(XX, YY)] = arr[(XX, YY)]
    return new_arr
    
def dilation(arr, stride, s_min, sift):
    arr = filter_pooling(arr, stride, s_min)
    arr = unpooling(arr, stride)
    arr = sifting(arr, sift=sift)
    return arr

class State:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0, v=0.0, quat=None, numpy_state=None):
        if numpy_state is not None:
            self.x = numpy_state[:, 0]
            self.y = numpy_state[:, 1]
            self.v = numpy_state[:, 2]
            self.yaw = numpy_state[:, 3]
        else:
            self.x = x
            self.y = y
            self.z = z
            self.yaw = yaw
            self.v = v

        self.quat = quat
        self.predelta = None            
    
    def distance(self, state):
        return np.linalg.norm([self.x - state.x, self.y - state.y, self.z - state.z])
    
    def to_array(self):
        return np.stack([self.x, self.y, self.v, self.yaw], axis=-1)
    
    def __repr__(self):
        info = f'\tx: {self.x:.2f}, y : {self.y:.2f}, z: {self.z:.2f}, v : {self.v:.2f}, yaw : {self.yaw:.2f}'
        return info
    
    def __add__(self, state):
        x = self.x + state.x
        y = self.y + state.y
        z = self.z + state.z
        yaw = self.yaw + state.yaw
        v = self.v + state.v        
        quat = MultiplyQuaternion(state.quat, self.quat).tolist()          
        return State(x, y, z, yaw, v, quat)      
    
    def __sub__(self, state):
        x = self.x - state.x
        y = self.y - state.y
        z = self.z - state.z
        yaw = self.yaw - state.yaw
        v = self.v - state.v                              
        quat = MultiplyQuaternion([state.quat[0],-state.quat[1],-state.quat[2],-state.quat[3]], self.quat).tolist()
        return State(x, y, z, yaw, v, quat)
    
    def __mul__(self, num):
        x = num * self.x
        y = num * self.y
        z = num * self.z
        yaw = num * self.yaw
        v = num * self.v
        quat = [self.quat[0], num*self.quat[1], num *self.quat[2], num*self.quat[3]]
        return State(x, y, z, yaw, v, quat)
    
    def __rmul__(self, num):
        x = num * self.x
        y = num * self.y
        z = num * self.z
        yaw = num * self.yaw
        v = num * self.v
        quat = [self.quat[0], num*self.quat[1], num *self.quat[2], num*self.quat[3]]
        return State(x, y, z, yaw, v, quat)    
        
def get_state(rob):
    Pos, Ang, linVel, angVel, Orn = rob.getState()
    v, w = rob.getRealVel(Ang)
    return State(x=Pos[0], y=Pos[1], z=Pos[2], v=v, yaw=pi_2_pi(Ang[2]), quat=Orn)

def pi_2_pi(angle):
    """
    Normalize an angle to [-pi, pi]
    """    
    if isinstance(angle, float):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi        
    else:
        angle[angle > np.pi] -= 2.0 * np.pi
        angle[angle < -np.pi] += 2.0 * np.pi
    return angle

def set_global_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)