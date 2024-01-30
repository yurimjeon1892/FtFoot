import numpy as np
from scipy.interpolate import interp1d

def compute_interp_path_from_wp(start_xp, start_yp, step=0.1):
    """
    Computes a reference path given a set of waypoints
    """
    final_xp = []
    final_yp = []
    delta = step  # [m]
    for idx in range(len(start_xp) - 1):
        section_len = np.sum(
            np.sqrt(
                np.power(np.diff(start_xp[idx : idx + 2]), 2)
                + np.power(np.diff(start_yp[idx : idx + 2]), 2)
            )
        )
        interp_range = np.linspace(0, 1, np.floor(section_len / delta).astype(int))
        fx = interp1d(np.linspace(0, 1, 2), start_xp[idx : idx + 2], kind=1)
        fy = interp1d(np.linspace(0, 1, 2), start_yp[idx : idx + 2], kind=1)
        # watch out to duplicate points!
        final_xp = np.append(final_xp, fx(interp_range)[1:])
        final_yp = np.append(final_yp, fy(interp_range)[1:])
    dx = np.append(0, np.diff(final_xp))
    dy = np.append(0, np.diff(final_yp))
    theta = np.arctan2(dy, dx)

    if len(final_xp) > 1:
        theta[0] = np.arctan2(final_yp[1] - final_yp[0], final_xp[1] - final_xp[0])
        return np.stack([final_xp, final_yp, theta], axis=-1)
    else:
        return []

def compute_path_from_wp(pts):
    """
    Computes a reference path given a set of waypoints
    """
    path = []
    for i in range(len(pts)):
        j = -1 if i == len(pts) - 1  else i + 1
        dx = pts[j][0] - pts[i][0]
        dy = pts[j][1] - pts[i][1]
        if j != -1:
            # theta = np.rad2deg(np.arctan2(dy, dx))
            theta = np.arctan2(dy, dx)
        path.append([pts[i][0], pts[i][1], theta])

    return np.array(path)