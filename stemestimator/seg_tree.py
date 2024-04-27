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
    tree = pclpy.pcl.search.KdTree.PointXYZ()
    cloud = pclpy.pcl.PointCloud.PointXYZ(points)

    normal_estimator = pclpy.pcl.features.NormalEstimationOMP.PointXYZ_Normal()
    normal_estimator.setInputCloud(cloud)
    normal_estimator.setSearchMethod(tree)
    normal_estimator.setRadiusSearch(search_radius)
    normals = pclpy.pcl.PointCloud.Normal()
    normal_estimator.compute(normals)
    return normals


def euclidean_cluster_extract(points, tolerance, min_cluster_size, max_cluster_size):
    """
    Takes a point cloud and clusters the points with euclidean clustering

    Args:
        points : np.ndarray
            (n,3) point cloud

        tolerance: int
            Maximum distance a point can be to a cluster to added to that cluster

        min_cluster_size: int
            Minimum number of points a cluster must have to be returned

        max_cluster_size: int
            Maximum number of points a cluster must have to be returned


    Returns:
        cluster_list : list
            List of (n,3) Pointclouds representing each cluster

    """
    filtered_points = pclpy.pcl.segmentation.EuclideanClusterExtraction.PointXYZ()
    kd_tree = pclpy.pcl.search.KdTree.PointXYZ()
    points_to_cluster = pclpy.pcl.PointCloud.PointXYZ(points)

    kd_tree.setInputCloud(points_to_cluster)
    filtered_points.setInputCloud(points_to_cluster)
    filtered_points.setClusterTolerance(tolerance)
    filtered_points.setMinClusterSize(min_cluster_size)
    filtered_points.setMaxClusterSize(max_cluster_size)
    filtered_points.setSearchMethod(kd_tree)

    point_indexes = pclpy.pcl.vectors.PointIndices()
    filtered_points.extract(point_indexes)

    cluster_list = [points_to_cluster.xyz[i2.indices] for i2 in point_indexes]
    return cluster_list


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


def segment_normals(
    points,
    search_radius=20,
    model=pclpy.pcl.sample_consensus.SACMODEL_LINE,
    method=pclpy.pcl.sample_consensus.SAC_RANSAC,
    normalweight=0.0001,
    miter=1000,
    distance=0.5,
    rlim=[0, 0.5],
):
    """
    Takes a point cloud and removes points that have less than minn neigbors in a certain radius

    Args:
        points : np.ndarray
            (n,3) point cloud

        search_radius: float
            Radius of the sphere a point can be in to be considered in the calculation of a sample points' normal

        model: int
            A pclpy.pcl.sample_consensus.MODEL value representing a ransac model

        method: float
            pclpy.pcl.sample_consensus.METHOD to use

        normalweight:
            Normal weight for ransacfromnormals

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
    pointcloud_normals = extract_normals(points, search_radius)

    pointcloud = pclpy.pcl.PointCloud.PointXYZ(points)
    segmenter = pclpy.pcl.segmentation.SACSegmentationFromNormals.PointXYZ_Normal()

    segmenter.setInputCloud(pointcloud)
    segmenter.setInputNormals(pointcloud_normals)
    segmenter.setDistanceThreshold(distance)
    segmenter.setOptimizeCoefficients(True)
    segmenter.setMethodType(method)
    segmenter.setModelType(model)
    segmenter.setMaxIterations(miter)
    segmenter.setRadiusLimits(rlim[0], rlim[1])
    segmenter.setDistanceFromOrigin(0.4)
    segmenter.setNormalDistanceWeight(normalweight)
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
            indices, model = segment_normals(p)
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
