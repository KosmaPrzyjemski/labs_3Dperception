import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from localisation import calculate_normal_vector

line_color = ['cyan', 'magenta', 'yellow', 'red', 'green', 'blue']
nb_segments = 5
iPick = 0
iClick = 0

def get_calib_params(intrinsic_filename, Rc_to_Rc2_filename):
    calibration_params = np.loadtxt(open(intrinsic_filename), delimiter=",")

    Rc_to_Rc2_matrix = np.loadtxt(open(Rc_to_Rc2_filename), delimiter=",")
    Rc2_to_Rc_matrix = np.linalg.inv(Rc_to_Rc2_matrix)

    [alphaU_l, alphaV_l, u0_l, v0_l] = calibration_params

    instrinsic_matrix = np.array([[alphaU_l, 0, u0_l],
                                       [0, alphaV_l, v0_l],
                                       [0, 0, 1]])

    return instrinsic_matrix, Rc_to_Rc2_matrix, Rc2_to_Rc_matrix


def select_segments():
    def on_click(event, clicked_points):
        global iClick
        if iClick >= nb_segments:
            return True

        u, v = event.xdata, event.ydata
        clicked_points.append([u, v, 1])

        if len(clicked_points) == 2:
            print("[*] Segment {:d}: p1({:.2f},{:.2f}) -- p2({:.2f},{:.2f})".format(
                iClick, clicked_points[0][0], clicked_points[0][1], clicked_points[1][0], clicked_points[1][1]))

            # plot the line
            ax2.plot([clicked_points[0][0], clicked_points[1][0]],
                     [clicked_points[0][1], clicked_points[1][1]],
                     color=line_color[iClick])

            # show the id of the edge
            ax2.text((clicked_points[0][0] + clicked_points[1][0]) / 2,
                     (clicked_points[0][1] + clicked_points[1][1]) / 2,
                     str(iClick),
                     color=line_color[iClick])

            iClick += 1

            # Calculate normal vector of the two segments formed by previously clicked points and R_c origin
            normal_vectors.append(calculate_normal_vector(clicked_points[0], clicked_points[1], intrinsic_matrix))

            print("[*] Normal vector = {}".format(normal_vectors[-1]))

            clicked_points[:] = []
            fig2.canvas.draw()

        return True

    def on_pick(event):
        global iPick
        global line_color
        if iPick >= nb_segments: return True

        if event.artist in lines:
            ind = lines.index(event.artist)
            lines[ind].set_color(line_color[iPick])
            ax1.text((X1_Ro[ind] + X2_Ro[ind]) / 2, (Y1_Ro[ind] + Y2_Ro[ind]) / 2, (Z1_Ro[ind] + Z2_Ro[ind]) / 2,
                     str(iPick), color=line_color[iPick])
            iPick += 1
            fig1.canvas.draw()
            pick_lines_Ro.append(model3D_Ro[ind, :])
        return True

    normal_vectors = []
    pick_lines_Ro = []
    # Get all points of the model expressed in Ro
    #   ---
    #   model3D_Ro: [Nx4] matrix, N being the total number of the 3D edges.
    #               Each row is a segment (or edge).
    #               Columns [0,1,2] describe the first 3D segment points.
    #               Columns [3,4,5] describe the second 3D segment points.
    #       [[x1, y1, z1, x2, y2, z2],
    #        [x1, y1, z1, x2, y2, z2],
    #                ...
    #        [x1, y1, z1, x2, y2, z2],
    #        [x1, y1, z1, x2, y2, z2]]
    model_points_3DRo = np.loadtxt("data/pikachu.xyz", dtype=float)
    model_edges = np.loadtxt("data/pikachu.edges", dtype=int)

    XYZ1_Ro = model_points_3DRo[model_edges[:, 0]]
    XYZ2_Ro = model_points_3DRo[model_edges[:, 1]]
    model3D_Ro = np.concatenate([XYZ1_Ro, XYZ2_Ro], axis=1)
    X1_Ro, Y1_Ro, Z1_Ro = XYZ1_Ro[:, 0], XYZ1_Ro[:, 1], XYZ1_Ro[:, 2]
    X2_Ro, Y2_Ro, Z2_Ro = XYZ2_Ro[:, 0], XYZ2_Ro[:, 1], XYZ2_Ro[:, 2]

    model_points_3DRo_final = np.loadtxt("data/pikachu_dino.xyz", dtype=float)
    model_edges_final = np.loadtxt("data/pikachu_dino.edges", dtype=int)
    XYZ1_Ro = model_points_3DRo_final[model_edges_final[:, 0]]
    XYZ2_Ro = model_points_3DRo_final[model_edges_final[:, 1]]
    model3D_Ro_final = np.concatenate([XYZ1_Ro, XYZ2_Ro], axis=1)


    # Retrieve all calibration parameters from YAML file
    #   ---
    #   intrinsic_matrix: [3x3] camera calibration matrix
    #       [[alpha_u,  0,          u0],
    #        [0,        alpha_v,    v0],
    #        [0,        0,          1]]
    #   ---
    #   Rc_to_Rc2_matrix: [4x4] Homogeneous transform from R_c_1 to R_c_2
    #   Rc2_to_Rc_matrix: [4x4] Homogeneous transform from R_c_2 to R_c_1
    intrinsic_matrix, Rc_to_Rc2_matrix, Rc2_to_Rc_matrix = \
        get_calib_params("data/calibration_parameters.txt", "data/Rc_to_Rc2_matrix.txt")

    # intrinsic_matrix = \
    #     get_calib_params("data/calibration_parameters.txt")

    # ----------------------------------------------------
    #               Segments selection
    # ----------------------------------------------------
    # /!\ Normal vectors for R_i --> R_c segments are
    # calculated during segment selection in the image /!\
    # See on_click(event) function.
    # ----------------------------------------------------
    # Read left/right/images
    image = mpimg.imread('data/image_1.JPG')
    image_2 = mpimg.imread('data/image_2.JPG')

    # Plot the model
    fig1 = plt.figure(1)
    ax1, lines = plot_3d_model(model3D_Ro, fig1)
    fig1.canvas.mpl_connect('pick_event', on_pick)  # Listen to mouse click event within figure 1
    plt.show(block=False)

    # Plot left image
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    plt.imshow(image)
    clicked_points = []
    fig2.canvas.mpl_connect('button_press_event',  # Listen to mouse click event within figure 2
                            lambda event: on_click(event, clicked_points))
    plt.show(block=False)

    # Wait for the edges to be selected
    while iPick < nb_segments or iClick < nb_segments:
        plt.pause(1)

    pick_lines_Ro = np.array(pick_lines_Ro)
    normal_vectors = np.array(normal_vectors)

    for idx in range(normal_vectors.shape[0]):
        ax1.plot([normal_vectors[idx, 0]], [normal_vectors[idx, 1]], [normal_vectors[idx, 2]], color=line_color[idx])

    return pick_lines_Ro, normal_vectors, image, image_2, model3D_Ro, model3D_Ro_final, intrinsic_matrix, \
                                                     Rc_to_Rc2_matrix

def plot_3d_model(model, fig):
    ax = Axes3D(fig)

    lines = []
    for idx in range(model.shape[0]):
        lines.append(ax.plot([model[idx, 0], model[idx, 3]],
                              [model[idx, 1], model[idx, 4]],
                              [model[idx, 2], model[idx, 5]],
                              color='k', picker=5)[0])
    ax.scatter(0, 0, 0, color='r', s=30)
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_zlabel('z (cm)')
    ax.set_title('3D model')

    return ax, lines


