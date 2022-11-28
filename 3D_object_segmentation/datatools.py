import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Constants
noise_sigma = .5                           # standard deviation error to be added
translation = 5.0                            # max translation of the test set
rotation = .8                               # max rotation (radians) of the test set

def rotation_matrix(axis, theta):
    '''
    Return a 3x3 rotation matrix
    Input:
        axis : 3x1 rotation vector
        theta: angle (in radians) of ratation around axis
    Output:
        3x3 rotation matix
    '''
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def disturb_data(data):
    '''
    Apply disturbances to the Nx3 data. This function uses the ofesets defined 
    in 'noise_sigma', 'translation' and 'rotation'.
    First, a different random translation between 0 and 'translation' is applied 
    to each point on each axis.
    Secondly, a random rotation of maximum 'rotation' angle is applied to the 
    intermediate resulting point cloud.
    Thirdly, a random noise is applied to each point.
    Finally, all points in the point cloud are shuffle to disrupt 
    correspondance between input and output data and point cloud is decimated 
    by removing 200 random points
    Input:
        data : Nx3 array of points
    Output:
        data : (N-200)x3 array of points
    '''
    B = np.copy(data)
    dim = data.shape[1]
    N = data.shape[0]
    

    # Translate
    t = np.random.rand(dim)*translation
    B += t

    # Rotate
    R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
    B = np.dot(R, B.T).T

    # Add noise
    B += np.random.randn(N, dim) * noise_sigma
    
    # Shuffle to disrupt correspondence
    np.random.shuffle(B)
    
    v=np.random.choice(B.shape[0], N-200, replace=False)
    return B[v,:]

def load_P_data_to_vec(filename, saveToXYZ=False, file=""):
    '''
    Load a point cloud from 3 pickle files (X.p, Y.p, Z.p) to a Nx3 array of points
    If specified, the loaded point cloud is saved into a ascii .xyz file
    Input:
        filename : path and root name of the pickle files
        saveToXYZ: boolean value whether loaded point cloud must be save or not
        file     : filename where point cloud must be saved if so
    Output:
        vec : loaded point cloud
    '''
    # Reading pickle files and storage in a vector
    fileX = os.path.join(filename + 'X.p');
    fileY = os.path.join(filename + 'Y.p');
    fileZ = os.path.join(filename + 'Z.p');
    if os.path.exists(fileX) and \
            os.path.exists(fileY) and \
            os.path.exists(fileZ):
        print("Opening pickles...")
        vec_X = pickle.load(open(fileX, 'rb'))
        vec_Y = pickle.load(open(fileY, 'rb'))
        vec_Z = pickle.load(open(fileZ, 'rb'))
        
        print(str(vec_X.shape[0]) + ' X points loaded')
        print(str(vec_Y.shape[0]) + ' Y points loaded')
        print(str(vec_Z.shape[0]) + ' Z points loaded')
        
        vec = np.array([vec_X, vec_Y, vec_Z]).T;
    
        if saveToXYZ:
            np.savetxt(file, vec, delimiter=' ');
        
        return vec;
    else:
        print("!!! Error - Unable to load files")
    
def load_XYZ_data_to_vec(filename):
    '''
    Load a point cloud from .xyz file to a Nx3 array of points
    Input:
        filename : path and root name of the pickle files
    Output:
        vec : loaded point cloud
    '''
    vec=np.loadtxt(filename, delimiter=' ');
    print(str(vec.shape) + ' points loaded');
    return vec;


def draw_data(data, title="", ax=""):
    '''
    Draws the point cloud given in data in the specified figure (ax) with the 
    specified title (title)
    Input:
        data : Nx3 array of points
        title: graph title
        ax   : figure where to draw the graph
    '''
    # Drawing 3D data
    ax.scatter(data[:,0], data[:,1], data[:,2], s=0.2, c='r')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    return


def draw_data_and_ref(data, ref="", title="", ax=""):
    '''
    Draws the 2 point clouds given in data and ref in the same specified figure
    (ax) with the specified title (title)
    Input:
        data : Nx3 array of points (in blue)
        ref  : Nx3 array of points (in red)
        title: graph title
        ax   : figure where to draw the graph
    '''
    # Affichage des donnees
    ax.scatter(data[:,0], data[:,1], data[:,2], s=0.2, c='b', marker='o')
    
    ax.scatter(ref[:,0], ref[:,1], ref[:,2], s=0.2, c='r', marker='o')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    return


