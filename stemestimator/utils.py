import pclpy
import open3d
import seg_tree
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

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

def rotation_matrix_from_vectors(vector1, vector2):
    """
        Finds a rotation matrix that can rotate vector1 to align with vector 2

        Args:
            vector1: np.narray (3)
                Vector we would apply the rotation to
        
            vector2: np.narray (3)
                Vector that will be aligned to

        Returns:
            rotation_matrix: np.narray (3,3)
                Rotation matrix that when applied to vector1 will turn it to the same direction as vector2
        """
    if all(np.abs(vector1)==np.abs(vector2)):
        return np.eye(3)
    a, b = (vector1 / np.linalg.norm(vector1)).reshape(3), (vector2 / np.linalg.norm(vector2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    matrix = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + matrix + matrix.dot(matrix) * ((1 - c) / (s ** 2))
    return rotation_matrix

def angle_between_vectors(vector1,vector2):
    """
        Finds the angle between 2 vectors

        Args:
            vec1: np.narray (3)
                First vector to measure angle from
        
            vec2: np.narray (3)
                Second vector to measure angle to

        Returns:
            None
        """
    value = np.sum(np.multiply(vector1, vector2)) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    if (value<-1) | (value>1):
        value = np.sign(value)
    angle = np.arccos(value)
    return angle

def makecylinder(model=[0,0,0,1,0,0,1],height = 1,density=10):
    """
        Makes a point cloud of a cylinder given a (7) parameter cylinder model and a length and density

        Args:
            model: np.narray (7)
                7 parameter cylinder model

            height: float
                Desired height of the generated cylinder

            density: int
                Desired density of the generated cylinder, 
                this density is determines the amount of points on each ring that composes the cylinder and on how many rings the cylinder will have

        Returns:
            rotated_cylinder: np.narray (n,3)
                3d point cloud of the desired cylinder
        """
    # extract info from cylinder model
    radius = model[6]
    X,Y,Z = model[:3]
    # get 3d points to make an upright cylinder centered to the origin
    n = np.arange(0,360,int(360/density))
    height = np.arange(0,height,height/density)
    n = np.deg2rad(n)
    x,z = np.meshgrid(n,height)
    x = x.flatten()
    z = z.flatten()
    cyl = np.vstack([np.cos(x)*radius,np.sin(x)*radius,z]).T
    # rotate and translate the cylinder to fit the model
    rotation = rotation_matrix_from_vectors([0,0,1],model[3:6])
    rotated_cylinder = np.matmul(rotation,cyl.T).T + np.array([X,Y,Z])
    return rotated_cylinder   

def plt3dpaint(nppoints, color_map = 'jet', reduce_for_vis = True, voxel_size = 0.2, pointsize = 0.1, subplots = 5):
    """
        displays point clouds on matplotlib 3d scatter plots

        Args:
            nppoints: pclpy.pcl.PointCloud.PointXYZRGB | pclpy.pcl.PointCloud.PointXYZ | np.ndarray | list | tuple
                Either a (n,3) point cloud or a list or tuple of point clouds to be displayed
            
            color_map: str | list 3
                By default uses jet color map, it can be a list with 3 ints between 0 and 255 to represent an RBG color to color all points

            reduce_for_vis: bool
                If true it performs voxel subsampling before displaying the point cloud

            voxel_size: float
                If reduce_for_vis is true, sets the voxel size for the voxel subsampling

            pointsize: int
                Size of the distplayed points

            subplots: int
                Number of subplots to create, each plot has a view rotation of 360/subplots

        Returns:
            None
        """
    assert (type(nppoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(nppoints) == pclpy.pcl.PointCloud.PointXYZ) or (type(nppoints) == np.ndarray) or (type(nppoints) is list) or (type(nppoints) is tuple), 'Not valid point_cloud'
    cloudlist = []
    cloudcolors = []
    if (type(nppoints) is not list) & (type(nppoints) is not tuple):
        nppoints = [nppoints]
        
    if len(nppoints) > 1:
        for n,i in enumerate(nppoints):
            workpoints = i
            if (type(workpoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(workpoints) == pclpy.pcl.PointCloud.PointXYZ):
                workpoints = workpoints.xyz

            if reduce_for_vis:
                workpoints = seg_tree.voxelize(workpoints,voxel_size)

            
            cloudmin = np.min(workpoints[:,2])
            cloudmax = np.max(workpoints[:,2])
    
            points = workpoints
            color_coef = n/len(nppoints)/2 + n%2*.5
            if type(color_map) == np.ndarray:
                color = color_map
            elif color_map == 'jet':
                color=cm.jet(color_coef)[:3]
            else:
                color=cm.Set1(color_coef)[:3]
            cloudcolors.append(np.ones_like(workpoints)*color + 0.4*(np.ones_like(workpoints) * ((workpoints[:,2] - cloudmin)/(cloudmax - cloudmin)).reshape(-1,1)-0.5) )
            cloudlist.append(points)
    else:
        workpoints = nppoints[0]
        if (type(workpoints) == pclpy.pcl.PointCloud.PointXYZRGB) or (type(workpoints) == pclpy.pcl.PointCloud.PointXYZ):
            workpoints = workpoints.xyz

        if reduce_for_vis:
            workpoints = seg_tree.voxelize(workpoints,voxel_size)
        cloudcolors.append(workpoints[:,2])
        cloudlist.append(workpoints)

    plt_pointcloud = np.concatenate(cloudlist)
    plt_colors = np.concatenate(cloudcolors)
    if len(nppoints) > 1:
        plt_colors = np.minimum(plt_colors,np.ones_like(plt_colors))
        plt_colors = np.maximum(plt_colors,np.zeros_like(plt_colors))
    fig = plt.figure(figsize=(30,16) )
    for i in range(subplots):
        ax = fig.add_subplot(1, subplots, i+1, projection='3d')
        ax.view_init(30, 360*i/subplots)
        ax.scatter3D(plt_pointcloud[:,0], plt_pointcloud[:,1], plt_pointcloud[:,2], c=plt_colors, s=pointsize)

def makesphere(centroid=[0, 0, 0], radius=1, dense=90):
    n = np.arange(0, 360, int(360 / dense))
    n = np.deg2rad(n)
    x, y = np.meshgrid(n, n)
    x = x.flatten()
    y = y.flatten()
    sphere = np.vstack(
        [
            centroid[0] + np.sin(x) * np.cos(y) * radius,
            centroid[1] + np.sin(x) * np.sin(y) * radius,
            centroid[2] + np.cos(x) * radius,
        ]
    ).T
    return sphere

def similarize(test, target):
    """
        Test a vectors angle to another vector and mirror its direction if it is greater than pi/2

        Args:
            test: np.narray (3)
                3d vector to test

            target: np.narray (3)
                3d vector to which test has to have an angle smaller than pi/2

        Returns:
            test: np.narray (3)
                3d vectors whos angle is below pi/2 with respect to the target vector
        """
    test = np.array(test)
    assert len(test) == 3,'vector must be dim 3'
    angle = angle_between_vectors(test,target)
    if angle > np.pi/2:
        test = -test
    return test

def Iscaled_dimensions(las_file, new_data):

    x_dimension = np.array(new_data['X'])
    scale = las_file.header.scales[0]
    offset = las_file.header.offsets[0]
    x = x_dimension + offset 

    y_dimension = np.array(new_data['Y'])
    offset = las_file.header.offsets[1]
    y = y_dimension + offset 

    z_dimension = np.array(new_data['Z'])
    offset = las_file.header.offsets[2]
    z = z_dimension + offset 
    return np.vstack([x, y, z]).T

def scaled_dimensions(las_file, ):
    xyz = las_file.xyz
    x_dimension = xyz[:,0]
    offset = las_file.header.offsets[0]
    x = (x_dimension  - offset)

    y_dimension = xyz[:,1]
    offset = las_file.header.offsets[1]
    y = (y_dimension  - offset)

    z_dimension = xyz[:,2]
    offset = las_file.header.offsets[2]
    z = (z_dimension  - offset)
    return np.vstack([x, y, z]).T
