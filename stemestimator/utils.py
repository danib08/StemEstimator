import cv2
import pclpy
import open3d
import seg_tree
import numpy as np
from matplotlib import cm

def open_3d_paint(nppoints, color_map='viridis', reduce_for_vis=False, voxel_size=0.1, point_size=0.1):
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
    :param point_size: the size of the points in the visualizer
    :type point_size: float
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
        options.background_color = np.asarray([0, 0, 0])
        options.point_size = point_size

        if len(nppoints) > 1:
            # If multiple point clouds are given, display them in different colors
            for n, i in enumerate(nppoints):
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

def get_stem_sections(points, num_sections=10):
    """Divides the stem point cloud into sections.

    :param points: The point cloud to be divided.
    :type points: np.ndarray (n,3)
    :param num_sections: The number of sections to divide the point cloud into.
    :type num_sections: int
    :return: The divided sections.
    :rtype: list
    """
    num_points = len(points)
    min_points_per_section = 5 

    if num_points < min_points_per_section:
        raise ValueError("Insufficient points to fit an ellipse.")

    # Sort points by their Z-coordinate
    points = points[points[:, 2].argsort()]

    # Determine the minimum section size based on the total number of points
    min_section_size = max(min_points_per_section, num_points // num_sections)

    sections = []
    start_idx = 0

    while start_idx < num_points:
        section_end = min(start_idx + min_section_size, num_points)
        section_points = points[start_idx:section_end]

        # Ensure each section has at least min_points_per_section points
        if len(section_points) < min_points_per_section:
            # If the remaining points are fewer than the minimum, include them in the current section
            section_points = points[start_idx:]

            if len(section_points) < min_points_per_section:
                # If the remaining points are still fewer than the minimum, add them to the last section
                if sections:
                    sections[-1] = np.vstack((sections[-1], section_points))
                else:
                    sections.append(section_points)
            break

        sections.append(section_points)
        start_idx = section_end

    return sections

#----------------------------------------------------------------------------------------------------
def fit_ellipse(points, bounding_box):
    ellipse = cv2.fitEllipse(points)
    (xc, yc), (d1, d2), angle = ellipse

    # Bounding limits
    x_min, y_min = bounding_box[0]
    x_max, y_max = bounding_box[1]

    if d1 > (x_max - x_min):
        d1 = x_max - x_min
    if d2 > (y_max - y_min):
        d2 = y_max - y_min

    adjusted_ellipse = ((xc, yc), (d1, d2), angle)
    semi_major_axis = d1 / 2
    semi_minor_axis = d2 / 2
    approximate_radius = (semi_major_axis + semi_minor_axis) / 2
    return adjusted_ellipse, approximate_radius

def generate_ellipse_points(ellipse, z, num_points=50):
    (xc, yc), (d1, d2), angle = ellipse
    t = np.linspace(0, 2 * np.pi, num_points)

    # Parametric equation of the ellipse
    X = (d1 / 2) * np.cos(t)
    Y = (d2 / 2) * np.sin(t)
    
    # Rotation matrix
    alpha = np.radians(angle)
    R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    ellipse_points = np.dot(R, np.array([X, Y]))
    X_rotated = ellipse_points[0, :] + xc
    Y_rotated = ellipse_points[1, :] + yc
    Z = np.full_like(X_rotated, z)
    return np.column_stack((X_rotated, Y_rotated, Z))

def visualize_results(stem_data, point_size=0.1):
    """
    Visualizes the stems with their fitted ellipses and radius annotations using Open3D.

    :param stem_data: List of dictionaries containing stem points, ellipse radii, 
    z coordinates, and ellipse points.
    :type stem_data: list of dicts
    :param point_size: the size of the points in the visualizer
    :type point_size: float
    """
    # Create a list to store the geometries for visualization
    geometries = []

    # Visualize stem points and ellipses
    for stem in stem_data:
        # Stem points
        stem_points = stem["stem_points"]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(stem_points)
        pcd.paint_uniform_color([0.1, 0.9, 0.1])  # Green color for stem points
        geometries.append(pcd)

        # Ellipse points
        ellipse_points = stem["ellipse_points"]
        for ellipse in ellipse_points:
            ellipse_pcd = open3d.geometry.PointCloud()
            ellipse_pcd.points = open3d.utility.Vector3dVector(ellipse)
            ellipse_pcd.paint_uniform_color([0.1, 0.1, 0.9])  # Blue color for ellipses
            geometries.append(ellipse_pcd)

    # Visualize all geometries
    visualizer = open3d.visualization.Visualizer()
    visualizer.create_window()
    for geom in geometries:
        visualizer.add_geometry(geom)
    
    options = visualizer.get_render_option()
    options.background_color = np.asarray([0, 0, 0])
    options.point_size = point_size
    
    visualizer.run()
    visualizer.destroy_window()    