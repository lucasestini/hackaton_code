import math
import numpy as np


def rotation_matrix(angle, direction, point=None):
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0,  0.0),
                     (0.0,  cosa, 0.0),
                     (0.0,  0.0,  cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array((( 0.0,         -direction[2],  direction[1]),
                      ( direction[2], 0.0,          -direction[0]),
                      (-direction[1], direction[0],  0.0)),
                     dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        print("Point is not none")
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotate(R,x):
    '''
    Returns the rotated 3-vector by rotation matrix R

    Input R is homogeneous matrix
    Input x is a 3-vector    
    '''
    x_ = np.append(x,[1])
    return R.dot(x_.reshape(4,))[:3]

def eucl(x,y):
    return np.sqrt((x[0]-y[0])**2 +(x[1]-y[1])**2 + (x[2]-y[2])**2 )


def build_handeyeMatrix():
    x = [270 * np.pi / 180, 90 * np.pi / 180, 10. * np.pi / 180, -13.3, 6.2,
         0]  # [4.7576930718362025, 1.9328924308040449, 0.45311419, -11.029967489936363, 4.656912437419396, 3.9550841767997666]#

    # Build an homogeneous transformation matrix from x
    phi = x[0]
    psi = x[1]
    x_ = np.sin(psi) * np.cos(phi)
    y_ = np.sin(psi) * np.sin(phi)
    z_ = np.cos(psi)
    T = rotation_matrix(x[2], [x_, y_, z_])
    T[:3, 3] = np.array([x[3], x[4], x[5]])


    T_left = np.copy(T)
    T_right = np.copy(T)
    T_right[0, 3] = -T_right[0, 3]
    T_right[0, 2] = -T_right[0, 2]
    T_right[2, 0] = -T_right[2, 0]
    return T_left, T_right
