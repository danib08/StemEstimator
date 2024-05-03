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

def cylinder_segmentation(points, max_iter=1000, search_radius=20, normal_weight=0.0001, 
                    max_distance=0.5, min_radius=0, max_radius=0.5):
    """Segments a point cloud into cylinder models using normals.

    :param points: The point cloud to be segmented.
    :type points: np.ndarray (n,3)
    :param max_iter: Maximum iterations for RANSAC.
    :type max_iter: int
    :param search_radius: The radius used to estimate the normals.
    :type search_radius: float
    :param normal_weight: The normal distance weight for RANSAC.
    :type normal_weight: float
    :param max_distance: Maximum distance a point can be from the model.
    :type max_distance: float
    :param min_radius: Minimum radius of the cylinder model.
    :type min_radius: float
    :param max_radius: Maximum radius of the cylinder model.
    :type max_radius: float
    :return: The indices of the points that fit the model and the model coefficients.
    :rtype: np.ndarray (n), np.ndarray (n)
    """
    point_cloud = pclpy.pcl.PointCloud.PointXYZ(points)
    point_cloud_normals = extract_normals(points, search_radius)

    method = pclpy.pcl.sample_consensus.SAC_RANSAC
    model = pclpy.pcl.sample_consensus.SACMODEL_CYLINDER
    segmenter = pclpy.pcl.segmentation.SACSegmentationFromNormals.PointXYZ_Normal()

    segmenter.setInputCloud(point_cloud)
    segmenter.setInputNormals(point_cloud_normals)

    segmenter.setModelType(model)
    segmenter.setMethodType(method)
    segmenter.setMaxIterations(max_iter)
    segmenter.setDistanceFromOrigin(0.4)
    segmenter.setOptimizeCoefficients(True)
    segmenter.setDistanceThreshold(max_distance)
    segmenter.setRadiusLimits(min_radius, max_radius)
    segmenter.setNormalDistanceWeight(normal_weight)

    point_indices = pclpy.pcl.PointIndices()
    coefficients = pclpy.pcl.ModelCoefficients()

    segmenter.segment(point_indices, coefficients)
    return point_indices.indices, coefficients.values

def region_growing(
    Points, ksearch=30, minc=20, maxc=100000, nn=30, smoothness=30.0, curvature=1.0
):
    """
    Takes a point cloud and clusters the points with region growing

    Args:
        points : np.ndarray
            (n,3) point cloud

        Ksearch: int
            Number of points used to estimate a points normal

        minc: int
            Minimum number of points a cluster must have to be returned

        maxc: int
            Maximum number of points a cluster must have to be returned

        nn: int
            Number of nearest neighbors used by the region growing algorithm

        smoothness:
            Smoothness threshold used in region growing

        curvature:
            Curvature threshold used in region growing

    Returns:
        region_growing_clusters: list
            list of (n,3) Pointclouds representing each cluster

    """
    pointcloud = pclpy.pcl.PointCloud.PointXYZ(Points)
    pointcloud_normals = pclpy.pcl.features.NormalEstimation.PointXYZ_Normal()
    tree = pclpy.pcl.search.KdTree.PointXYZ()

    pointcloud_normals.setInputCloud(pointcloud)
    pointcloud_normals.setSearchMethod(tree)
    pointcloud_normals.setKSearch(ksearch)
    normals = pclpy.pcl.PointCloud.Normal()
    pointcloud_normals.compute(normals)

    region_growing_clusterer = pclpy.pcl.segmentation.RegionGrowing.PointXYZ_Normal()
    region_growing_clusterer.setInputCloud(pointcloud)
    region_growing_clusterer.setInputNormals(normals)
    region_growing_clusterer.setMinClusterSize(minc)
    region_growing_clusterer.setMaxClusterSize(maxc)
    region_growing_clusterer.setSearchMethod(tree)
    region_growing_clusterer.setNumberOfNeighbours(nn)
    region_growing_clusterer.setSmoothnessThreshold(smoothness / 180.0 * np.pi)
    region_growing_clusterer.setCurvatureThreshold(curvature)

    clusters = pclpy.pcl.vectors.PointIndices()
    region_growing_clusterer.extract(clusters)

    region_growing_clusters = [pointcloud.xyz[i2.indices] for i2 in clusters]
    return region_growing_clusters


def segment(
    points,
    model=pclpy.pcl.sample_consensus.SACMODEL_LINE,
    method=pclpy.pcl.sample_consensus.SAC_RANSAC,
    miter=1000,
    distance=0.5,
    rlim=[0, 0.5],
):
    """
    Takes a point cloud and removes points that have less than minn neigbors in a certain radius

    Args:
        points : np.ndarray
            (n,3) point cloud

        model: int
            A pclpy.pcl.sample_consensus.MODEL value representing a ransac model

        method: float
            pclpy.pcl.sample_consensus.METHOD to use

        miter: bool
            Maximum iterations for ransac

        distance:
            Maximum distance a point can be from the model

        rlim:
            Radius limit for cylinder model


    Returns:
        pI.indices: np.narray (n)
            Indices of points that fit the model

        Mc.values: np.narray (n)
            Model coefficients

    """
    pointcloud = pclpy.pcl.PointCloud.PointXYZ(points)
    segmenter = pclpy.pcl.segmentation.SACSegmentation.PointXYZ()

    segmenter.setInputCloud(pointcloud)
    segmenter.setDistanceThreshold(distance)
    segmenter.setOptimizeCoefficients(True)
    segmenter.setMethodType(method)
    segmenter.setModelType(model)
    segmenter.setMaxIterations(miter)
    segmenter.setRadiusLimits(rlim[0], rlim[1])
    pI = pclpy.pcl.PointIndices()
    Mc = pclpy.pcl.ModelCoefficients()
    segmenter.segment(pI, Mc)
    return pI.indices, Mc.values

def findstemsLiDAR(pointsXYZ):
    """
    Takes a point cloud from a Cylindrical LiDAR and extract stems and their models

    Args:
        points : np.ndarray
            (n,3) point cloud

    Returns:
        stemsR : list(np.narray (n,3))
            List of (n,3) Pointclouds belonging to each stem

        models : list(np.narray (n))
            List of model coefficients corresponding to each extracted stem

    """
    non_ground_points, ground = remove_ground(pointsXYZ)
    flatpoints = np.hstack(
        [non_ground_points[:, 0:2], np.zeros_like(non_ground_points)[:, 0:1]]
    )

    filtered_points = radius_outlier_removal(flatpoints)
    notgoodpoints = non_ground_points[np.isnan(filtered_points[:, 0])]
    goodpoints = non_ground_points[np.bitwise_not(np.isnan(filtered_points[:, 0]))]

    cluster_list = euclidean_cluster_extract(goodpoints)
    rg_clusters = []
    for i in cluster_list:
        rg_clusters.append(region_growing(i))

    models = []
    stem_clouds = []
    for i in rg_clusters:
        for p in i:
            indices, model = cylinder_segmentation(p)
            prop = len(p[indices]) / len(p)
            if (
                len(indices) > 1
                and prop > 0.0
                and np.arccos(np.dot([0, 0, 1], model[3:6])) < 0.6
            ):
                points = p[indices]
                PC, _, _ = Plane.getPrincipalComponents(points)
                if PC[0] / PC[1] > 10:
                    stem_clouds.append(points)
                    models.append(model)
    return stem_clouds, models

def box_crop(points, min, max):
    if type(points) == pclpy.pcl.PointCloud.PointXYZ:
        sub_pcd = pclpy.pcl.PointCloud.PointXYZ()
        cropfilter = pclpy.pcl.filters.CropBox.PointXYZ()
    elif pclpy.pcl.PointCloud.PointXYZRGB:
        sub_pcd = pclpy.pcl.PointCloud.PointXYZRGB()
        cropfilter = pclpy.pcl.filters.CropBox.PointXYZ()
    cropfilter.setMin(np.asarray(min))
    cropfilter.setMax(np.asarray(max))
    cropfilter.setInputCloud(points)
    cropfilter.filter(sub_pcd)
    return sub_pcd.xyz
