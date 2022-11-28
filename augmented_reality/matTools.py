import numpy as np


def construct_matrix_from_vec(vec_solution):
    a = vec_solution[0]
    b = vec_solution[1]
    g = vec_solution[2]
    x = vec_solution[3]
    y = vec_solution[4]
    z = vec_solution[5]

    matPos = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])

    matPos[0, 0] = np.cos(g) * np.cos(b)
    matPos[0, 1] = np.cos(g) * np.sin(b) * np.sin(a) - np.sin(g) * np.cos(a)
    matPos[0, 2] = np.cos(g) * np.sin(b) * np.cos(a) + np.sin(g) * np.sin(a)
    matPos[0, 3] = x

    matPos[1, 0] = np.sin(g) * np.cos(b)
    matPos[1, 1] = np.sin(g) * np.sin(b) * np.sin(a) + np.cos(g) * np.cos(a)
    matPos[1, 2] = np.sin(g) * np.sin(b) * np.cos(a) - np.cos(g) * np.sin(a)
    matPos[1, 3] = y

    matPos[2, 0] = -np.sin(b)
    matPos[2, 1] = np.cos(b) * np.sin(a)
    matPos[2, 2] = np.cos(b) * np.cos(a)
    matPos[2, 3] = z
    return matPos


def construct_vec_from_matrix(extrinsic):
    soluL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    sb = -extrinsic[2, 0]

    # if b = pi/2
    if np.abs(sb) >= 0.99999:
        if sb >= 0.0:
            soluL[1] = np.pi / 2.0
            aMoinsg = np.arctan2(extrinsic[1, 1], extrinsic[0, 1])
            aPlusg = np.arctan2(extrinsic[1, 2], extrinsic[1, 2])
            soluL[0] = (aMoinsg + aPlusg) / 2.0
            soluL[2] = soluL[0] - aMoinsg
        else:
            soluL[1] = -np.pi / 2.0
            aMoinsg = np.arctan2(extrinsic[1, 1], -extrinsic[0, 1])
            aPlusg = np.arctan2(extrinsic[1, 2], -extrinsic[1, 2])
            soluL[0] = (aMoinsg + aPlusg) / 2.0
            soluL[2] = soluL[0] - aMoinsg
    else:
        cb = np.sqrt(1.0 - sb * sb)
        sa = extrinsic[2, 1] / cb
        ca = extrinsic[2, 2] / cb
        sg = extrinsic[1, 0] / cb
        cg = extrinsic[0, 0] / cb
        soluL[0] = np.arctan2(sa, ca)
        soluL[1] = np.arctan2(sb, cb)
        soluL[2] = np.arctan2(sg, cg)
    if soluL[0] > np.pi: soluL[0] = soluL[0] - 2.0 * np.pi
    if soluL[1] > np.pi: soluL[1] = soluL[1] - 2.0 * np.pi
    if soluL[2] > np.pi: soluL[2] = soluL[2] - 2.0 * np.pi
    soluL[3] = extrinsic[0, 3]
    soluL[4] = extrinsic[1, 3]
    soluL[5] = extrinsic[2, 3]

    return soluL


def transform_point_with_matrix(transformation_matrix, initial_point):
    initial_point_cpy = np.ones((initial_point.shape[0], 4))
    initial_point_cpy[:, 0:3] = np.copy(initial_point)

    transformed_point = np.dot(transformation_matrix, initial_point_cpy.T)

    return transformed_point[0:3, :].T
