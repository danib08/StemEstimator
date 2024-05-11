import pclpy
import open3d
import seg_tree
import numpy as np
from matplotlib import cm

def open_3d_paint(nppoints, color_map='jet', reduce_for_vis=False, voxel_size=0.1, pointsize=0.1):
    """
    Opens an open3d visualizer and displays point clouds.

    :param nppoints: the point cloud(s) to be displayed
    :type nppoints: pclpy.pcl.PointCloud.PointXYZ or np.ndarray or list or tuple
    :param color_map: the color map to use for the point cloud
    :type color_map: str or list
    :param reduce_for_vis: whether to reduce the point cloud density for visualization
    :type reduce_for_vis: bool
    :param voxel_size: the voxel size in case of point cloud reduction
    :type voxel_size: float
    :param pointsize: the size of the points in the visualizer
    :type pointsize: int
    :raises ValueError: if the type of nppoints is invalid.
    :raises Exception: if an error occurs during visualization.
    :return: None
    """
    valid_types = [pclpy.pcl.PointCloud.PointXYZ, np.ndarray, list, tuple]
    if not any(isinstance(nppoints, t) for t in valid_types):
        raise ValueError("Invalid type for 'nppoints'. It should be one of: pclpy.pcl.PointCloud.PointXYZ, "
                         "np.ndarray, list, tuple.")
    
    if not isinstance(nppoints, (list, tuple)):
        nppoints = [nppoints]
    try:
        visualizer = open3d.visualization.Visualizer()
        visualizer.create_window()
        options = visualizer.get_render_option()
        options.background_color = np.asarray([1, 1, 1])
        options.point_size = pointsize

        if len(nppoints) > 1:
            # If multiple point clouds are given, display them in different colors
            for n,i in enumerate(nppoints):
                workpoints = i
                if (type(workpoints) == pclpy.pcl.PointCloud.PointXYZ):
                    workpoints = workpoints.xyz

                if reduce_for_vis:
                    workpoints = seg_tree.voxelize(workpoints, voxel_size)

                points = convert_cloud(workpoints)
                color_coef = n/len(nppoints)/2 + n%2*.5

                if type(color_map) == np.ndarray:
                    color = color_map
                elif color_map == 'jet':
                    color = cm.jet(color_coef)[:3]
                else:
                    color = cm.Set1(color_coef)[:3]

                points.colors = open3d.utility.Vector3dVector(np.ones_like(workpoints)*color)
                visualizer.add_geometry(points)
        else:
            workpoints = nppoints[0]
            if (type(workpoints) == pclpy.pcl.PointCloud.PointXYZ):
                workpoints = workpoints.xyz
                
            if reduce_for_vis:
                workpoints = seg_tree.voxelize(workpoints, voxel_size)

            points = convert_cloud(workpoints)
            visualizer.add_geometry(points)

        visualizer.run()
        visualizer.destroy_window()
        
    except Exception as e:
        print(type(e))
        print(e.args)
        print(e)
        visualizer.destroy_window()
        
def convert_cloud(points):
    """
    Converts a numpy array point cloud to an open3d point cloud.

    :param points: the point cloud to be converted
    :type points: np.ndarray (n,3)
    :return: the Open3D point cloud
    :rtype: open3d.geometry.PointCloud
    """
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    return pcd

def get_principal_vectors(A):
    """Get the principal vectors and values of a matrix centered around (0,0,0)

    :param A: The matrix to get the principal vectors and values from.
    :type A: np.ndarray (n,m)
    :return: The principal vectors and values in descending order.
    :rtype: np.ndarray (n,m), np.ndarray (n,m)
    """
    eigenvalues, eigenvectors = np.linalg.eig(np.matmul(A.T, A))
    sorted_indices = np.argsort(eigenvalues)[::-1]
    values = eigenvalues[sorted_indices]
    vectors = eigenvectors[:, sorted_indices]
    return vectors, values

def distance_point_to_line(point, line_point1, line_point2=np.array([0,0,0])):
    """Get the distance from a point to a line.

    :param point: The point to measure the distance from.
    :type point: np.ndarray (3)
    :param line_point1: The first point of the line.
    :type line_point1: np.ndarray (3)
    :param line_point2: The second point of the line.
    :type line_point2: np.ndarray (3)
    :return: The distance from the point to the line.
    :rtype: float
    """
    vector1 = point - line_point1
    vector2 = point - line_point2
    distance = np.linalg.norm(np.cross(vector2, vector1))
    normalized = distance / np.linalg.norm(line_point1 - line_point2)
    return normalized

def angle_between_vectors(vector1, vector2):
    """Get the angle between two vectors.

    :param vector1: The first vector.
    :type vector1: np.ndarray (3)
    :param vector2: The second vector.
    :type vector2: np.ndarray (3)
    :return: The angle between the two vectors.
    :rtype: float
    """
    dot_product = np.dot(vector1, vector2)
    vector1_norm = np.linalg.norm(vector1)
    vector2_norm = np.linalg.norm(vector2)
    
    value = dot_product / (vector1_norm * vector2_norm)   
    value = np.clip(value, -1, 1)

    angle = np.arccos(value)
    return angle

def similarize(test, target):
    """Make two vectors point in the same direction.

    :param test: The vector to be aligned.
    :type test: np.ndarray (3)
    :param target: The vector to align to.
    :type target: np.ndarray (3)
    :return: The aligned vector.
    :rtype: np.ndarray (3)
    """
    test = np.array(test)
    target = np.array(target)
    angle = angle_between_vectors(test, target)

    if angle > np.pi/2:
        test = -test
    return test

def make_cylinder(model, heights=1, density=10):
    """Create a cylinder point cloud based on a model.

    :param model: The model to create the cylinder from.
    :type model: np.ndarray (7)
    :param height: The height of the cylinder.
    :type height: float
    :param density: The density of the cylinder.
    :type density: int
    :return: The cylinder point cloud.
    :rtype: np.ndarray (n,3)
    """
    radius = model[6]
    X, Y, Z = model[:3]
    direction_vector = model[3:6]

    # Get 3D points to make an upright cylinder centered to the origin
    angles = np.arange(0, 360, int(360/density))
    heights = np.arange(0, heights, heights/density)
    angles = np.deg2rad(angles)

    x, z = np.meshgrid(angles, heights)
    x = x.flatten()
    z = z.flatten() 

    cyl = np.vstack([np.cos(x)*radius, np.sin(x)*radius, z]).T
    # Rotate and translate the cylinder to fit the model
    rotation = rotation_matrix_from_vectors([0,0,1], direction_vector)
    rotated_cylinder = np.matmul(rotation, cyl.T).T + np.array([X,Y,Z])
    return rotated_cylinder   

def rotation_matrix_from_vectors(vector1, vector2):
    """Finds a rotation matrix that can rotate the first vector to align with second.
    
    :param vector1: The vector to be rotated.
    :type vector1: np.ndarray (3)
    :param vector2: The vector to align to.
    :type vector2: np.ndarray (3)
    :return: The rotation matrix.
    :rtype: np.ndarray (3,3)
    """
    if all(np.abs(vector1) == np.abs(vector2)):
        return np.eye(3)
    
    # normalizing the vectors
    a, b = (vector1 / np.linalg.norm(vector1)).reshape(3), (vector2 / np.linalg.norm(vector2)).reshape(3)
    
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    matrix = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + matrix + matrix.dot(matrix) * ((1 - c) / (s ** 2))
    return rotation_matrix