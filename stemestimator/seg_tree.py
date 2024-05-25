import pclpy
import numpy as np
import open3d as o3d

def voxelize(points, voxel_size, use_o3d=False):
    """Uses voxel grid filtering to downsample a point cloud.

    :param points: The point cloud to be downsampled.
    :type points: np.ndarray (n,3)
    :param voxel_size: The size of the voxel grid.
    :type voxel_size: float
    :param use_o3d: Whether to use Open3D for the voxel grid filtering.
    :type use_o3d: bool
    :return: The downsampled point cloud.
    :rtype: np.ndarray (n,3)
    """
    if use_o3d:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downsized_pcd = pcd.voxel_down_sample(voxel_size)
        return np.array(downsized_pcd.points)
    else:
        cloud = pclpy.pcl.PointCloud.PointXYZ(points)
        voxel_filter = pclpy.pcl.filters.VoxelGrid.PointXYZ()
        filtered_pointcloud = pclpy.pcl.PointCloud.PointXYZ()

        voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size)
        voxel_filter.setInputCloud(cloud)

        voxel_filter.filter(filtered_pointcloud)
        if type(points) == pclpy.pcl.PointCloud.PointXYZRGB:
            return filtered_pointcloud
        else:
            return filtered_pointcloud.xyz

def remove_ground(point_cloud, max_window_size=20, slope=1.0, 
                 initial_distance=0.5, max_distance=3.0):
    """Applies the Progressive Morphological Filter to a point cloud in order to 
    separate the ground points from the non-ground points.

    :param point_cloud: The point cloud to be filtered.
    :type point_cloud: np.ndarray (n,3)
    :param max_window_size: The maximum window size to be used in filtering ground returns.
    :type max_window_size: int
    :param slope: Slope value to be used in computing the height threshold.
    :type slope: float
    :param initial_distance: Initial height above the parameterized ground surface to be considered a ground return.
    :type initial_distance: float
    :param max_distance: Maximum height above the parameterized ground surface to be considered a ground return.
    :type max_distance: float
    :return: The non-ground and ground point clouds.
    :rtype: np.ndarray (n,3), np.ndarray (n,3)
    """
    ground_indices = pclpy.pcl.vectors.Int()

    # Apply the Progressive Morphological Filter
    filter = pclpy.pcl.segmentation.ApproximateProgressiveMorphologicalFilter.PointXYZ()
    filter.setInputCloud(point_cloud)
    filter.setMaxWindowSize(max_window_size)
    filter.setSlope(slope)
    filter.setInitialDistance(initial_distance)
    filter.setMaxDistance(max_distance)
    filter.extract(ground_indices)

    # Separate the ground and non-ground points
    ground = pclpy.pcl.PointCloud.PointXYZ()
    non_ground = pclpy.pcl.PointCloud.PointXYZ()
    extract = pclpy.pcl.filters.ExtractIndices.PointXYZ()
    extract.setInputCloud(point_cloud)
    extract.setIndices(ground_indices)
    extract.filter(ground)
    extract.setNegative(True)
    extract.filter(non_ground)

    return non_ground.xyz, ground.xyz

def radius_outlier_removal(points, min_n=6, radius=0.4, organized=True):
    """Removes points that have less than min_n neighbors in a certain radius.

    :param points: The point cloud to be filtered.
    :type points: np.ndarray (n,3)
    :param min_n: The minimum number of neighbors a point must have to be kept.
    :type min_n: int
    :param radius: The radius within which to search for neighbors.
    :type radius: float
    :param organized: Whether outlier points are set to NaN instead of removing the points from the cloud. q
    :type organized: bool
    :return: The filtered point cloud.
    :rtype: np.ndarray (n,3)
    """
    cloud = pclpy.pcl.PointCloud.PointXYZ(points)

    ror_filter = pclpy.pcl.filters.RadiusOutlierRemoval.PointXYZ()
    ror_filter.setInputCloud(cloud)
    ror_filter.setMinNeighborsInRadius(min_n)
    ror_filter.setRadiusSearch(radius)
    ror_filter.setKeepOrganized(organized)

    filtered_point_cloud = pclpy.pcl.PointCloud.PointXYZ()
    ror_filter.filter(filtered_point_cloud)
    return filtered_point_cloud.xyz

def extract_normals(points, search_radius):
    """Estimates the normals of a point cloud using OpenMP approach.

    :param points: The point cloud.
    :type points: np.ndarray (n,3)
    :param search_radius: The radius used to estimate the normals.
    :type search_radius: float
    :return: Normal vectors corresponding to the points in the input point cloud.
    :rtype: np.ndarray (n,3)
    """
    kd_tree = pclpy.pcl.search.KdTree.PointXYZ()
    cloud = pclpy.pcl.PointCloud.PointXYZ(points)

    normal_estimator = pclpy.pcl.features.NormalEstimationOMP.PointXYZ_Normal()
    normal_estimator.setInputCloud(cloud)
    normal_estimator.setSearchMethod(kd_tree)
    normal_estimator.setRadiusSearch(search_radius)
    normals = pclpy.pcl.PointCloud.Normal()
    normal_estimator.compute(normals)
    return normals

def euclidean_cluster_extract(points, tolerance, min_cluster_size, max_cluster_size):
    """Clusters points based on Euclidean distance.

    :param points: The point cloud to be clustered.
    :type points: np.ndarray (n,3)
    :param tolerance: The maximum distance between two points to be considered in the same cluster.
    :type tolerance: float
    :param min_cluster_size: The minimum number of points a cluster must have to be returned.
    :type min_cluster_size: int
    :param max_cluster_size: The maximum number of points a cluster must have to be returned.
    :type max_cluster_size: int
    :return: The clusters of points.
    :rtype: list(np.ndarray (n,3))
    """
    points_to_cluster = pclpy.pcl.PointCloud.PointXYZ(points)
    kd_tree = pclpy.pcl.search.KdTree.PointXYZ()
    clustering = pclpy.pcl.segmentation.EuclideanClusterExtraction.PointXYZ()

    kd_tree.setInputCloud(points_to_cluster)
    clustering.setInputCloud(points_to_cluster)
    clustering.setClusterTolerance(tolerance)
    clustering.setMinClusterSize(min_cluster_size)
    clustering.setMaxClusterSize(max_cluster_size)
    clustering.setSearchMethod(kd_tree)

    cluster_indices = pclpy.pcl.vectors.PointIndices()
    clustering.extract(cluster_indices)

    cluster_list = [points_to_cluster.xyz[cluster.indices] for cluster in cluster_indices]
    return cluster_list