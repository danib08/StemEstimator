import pclpy
import utils
import seg_tree
import numpy as np
from ellipse import LsqEllipse

class TreeTool:
    """
    Holds all methods to process point cloud into a list of all tree stem locations and DBHs.

    :param point_cloud: The 3D point cloud of the forest to be processed.
    :type point_cloud: np.ndarray (n,3) or pclpy.pcl.PointCloud.PointXYZ
    """
    def __init__(self, point_cloud=None):
        """Constructor method. Initializes the point cloud and the attributes to store the results.

        :raises ValueError: If the point cloud is not of the correct type.
        """
        valid_types = [pclpy.pcl.PointCloud.PointXYZ, np.ndarray, None]

        if point_cloud is not None and type(point_cloud) not in valid_types:
            raise ValueError("Invalid type for 'point_cloud'. It should be one of: "
                             "pclpy.pcl.PointCloud.PointXYZ, np.ndarray.")
        
        self.point_cloud = None
        self.non_ground_cloud = None
        self.ground_cloud = None

        if point_cloud is not None:
            if isinstance(point_cloud, np.ndarray):
                self.point_cloud = pclpy.pcl.PointCloud.PointXYZ(point_cloud)
            else:
                self.point_cloud = point_cloud

    def step_1_remove_ground(self, max_window_size=20, slope=1.0, initial_distance=0.5, max_distance=3.0):
        """Removes the ground points from the point cloud using a filter.
        :param max_window_size: The maximum window size for the filter.
        :type max_window_size: int
        :param slope: The slope of the filter.
        :type slope: float
        :param initial_distance: The initial distance for the filter.
        :type initial_distance: float
        :param max_distance: The maximum distance for the filter.
        :type max_distance: float
        :return: None
        """
        non_ground, ground = seg_tree.remove_ground(self.point_cloud, max_window_size, slope, 
                                                    initial_distance, max_distance)
        self.non_ground_cloud = pclpy.pcl.PointCloud.PointXYZ(non_ground)
        self.ground_cloud = pclpy.pcl.PointCloud.PointXYZ(ground)

    def step_2_normal_filtering(self, search_radius, verticality_threshold, curvature_threshold, min_points=0):
        """Filters the non-ground points based on their normals, removing vertical and curved points.

        :param search_radius: Radius used for neighborhood search to calculate normals
        :type search_radius: float
        :param verticality_threshold: Threshold used to filter out points based on their verticality.
        :type verticality_threshold: float
        :param curvature_threshold: Threshold used to filter out points based on their curvature.
        :type curvature_threshold: float
        :param min_points: Minimum number of points required for neighborhood search (optional).
        :type min_points: int
        """
        if min_points > 0:
            subject_cloud = seg_tree.radius_outlier_removal(self.non_ground_cloud.xyz, min_points, 
                                                            search_radius, organized=False)
        else:
            subject_cloud = self.non_ground_cloud.xyz

        non_ground_normals = seg_tree.extract_normals(subject_cloud, search_radius)

        # Removing NaN points
        non_nan_mask = np.bitwise_not(np.isnan(non_ground_normals.normals[:, 0]))
        non_nan_cloud = subject_cloud[non_nan_mask]
        non_nan_normals = non_ground_normals.normals[non_nan_mask]
        non_nan_curvature = non_ground_normals.curvature[non_nan_mask]

        # Filtering by verticality and curvature
        verticality = np.dot(non_nan_normals, [[0], [0], [1]]) # Dot product with vertical vector
        verticality_mask = (verticality < verticality_threshold) & (-verticality_threshold < verticality)

        curvature_mask = non_nan_curvature < curvature_threshold
        verticality_curvature_mask = verticality_mask.ravel() & curvature_mask.ravel()

        only_horizontal_points = non_nan_cloud[verticality_curvature_mask]
        only_horizontal_normals = non_nan_normals[verticality_curvature_mask]

        # Set filtered and non filtered points
        self.non_ground_normals = non_ground_normals
        self.non_filtered_normals = non_nan_normals
        self.non_filtered_points = pclpy.pcl.PointCloud.PointXYZ(non_nan_cloud)
        self.filtered_points = pclpy.pcl.PointCloud.PointXYZ(only_horizontal_points) # Trunk points
        self.filtered_normals = only_horizontal_normals

    def step_3_euclidean_clustering(self, tolerance, min_cluster_size, max_cluster_size):
        """Clusters the normal-filtered points using euclidean distance 
        and assigns them to attribute cluster_list.

        :param tolerance: The tolerance used for clustering.
        :type tolerance: float
        :param min_cluster_size: The minimum number of points required for a cluster.
        :type min_cluster_size: int
        :param max_cluster_size: The maximum number of points allowed for a cluster.
        :type max_cluster_size: int
        :return: None
        """
        self.cluster_list = seg_tree.euclidean_cluster_extract(self.filtered_points.xyz, 
                                                               tolerance, min_cluster_size, max_cluster_size)
        
    def step_4_group_stems(self, max_distance=0.4):
        """Groups the clusters of points into stems based on their centroids and principal directions.

        :param max_distance: The maximum distance allowed between a point and the line formed by 
        the first principal vector of another cluster.
        :type max_distance: float
        :return: None
        """
        stem_groups = []
        for cluster in self.cluster_list:
            centroid = np.mean(cluster, axis=0)
            vectors, values = utils.get_principal_vectors(cluster - centroid)
            # Straightness ratio gives an indication of how aligned the points are along
            # the principal direction defined by the largest eigenvalue.
            straightness = values[0] / (values[0] + values[1] + values[2])

            clusters_dict = {
                "cloud": cluster,
                "straightness": straightness,
                "centroid": centroid,
                "principal_vectors": vectors
            }
            stem_groups.append(clusters_dict)

        # For each cluster, test if its centroid is near the line formed by the first principal vector 
        # of another cluster parting from the centroid of that cluster. If so, join the two clusters.
        temp_stems = [i["cloud"] for i in stem_groups]
        num_clusters = len(temp_stems)

        for tree1 in reversed(range(0, num_clusters)):
            for tree2 in reversed(range(0, tree1)):
                centroid1 = stem_groups[tree1]["centroid"]
                centroid2 = stem_groups[tree2]["centroid"]

                if np.linalg.norm(centroid1[:2] - centroid2[:2]) < 2:
                    vector1 = stem_groups[tree1]["principal_vectors"][0]
                    vector2 = stem_groups[tree2]["principal_vectors"][0]
                    dist1 = utils.distance_point_to_line(centroid2, vector1 + centroid1, centroid1)
                    dist2 = utils.distance_point_to_line(centroid1, vector2 + centroid2, centroid2)
                    
                    if (dist1 < max_distance) | (dist2 < max_distance):
                        temp_stems[tree2] = np.vstack([temp_stems[tree2], temp_stems.pop(tree1)])
                        break

        self.stem_groups = stem_groups
        self.complete_stems = temp_stems

    def step_5_get_ground_level_trees(self, lowstems_height=5):
        """Filters stems to only keep those near the ground and crops them up to a certain height.

        :param lowstems_height: The height threshold for low stems.
        :type lowstems_height: int
        :return: None
        """
        # Generate a bivariate quadratic equation to model the ground
        ground_points = self.ground_cloud.xyz
        coefficient_matrix = np.c_[
            np.ones(ground_points.shape[0]),
            ground_points[:, :2],
            np.prod(ground_points[:, :2], axis=1),
            ground_points[:, :2] ** 2,
        ]
        self.ground_model_c, _, _, _ = np.linalg.lstsq(coefficient_matrix, ground_points[:, 2], rcond=None)

        # Obtains a ground point for each stem by taking the XY component of the centroid
        # and obtaining the coresponding Z coordinate from the quadratic ground model
        self.stems_with_ground = []
        for i in self.complete_stems:
            centroid = np.mean(i, 0)
            X, Y = centroid[:2]
            Z = np.dot(
                    np.c_[np.ones(X.shape), X, Y, X * Y, X**2, Y**2],
                    self.ground_model_c,)

            self.stems_with_ground.append([i, [X, Y, Z[0]]])

        # Filter stems that do not have points below our lowstems_height threshold
        low_stems = [
            i
            for i in self.stems_with_ground
            if np.min(i[0], axis=0)[2] < (lowstems_height + i[1][2])
        ]
        
        """
        NOTE: each stem is a list with two elements, the first element is the point cloud 
        of the stem and the second element is the centroid of the stem.
        [stem_points, [X, Y, Z]]
        """
        self.low_stems = low_stems
        self.low_stems_visualize = [i[0] for i in low_stems]

    def step_6_fit_ellipses(self):
        final_stems = []
        visualization_ellipses = []

        for stem_points, _ in self.low_stems:
            ellipse_diameters = []
            section_list = utils.get_stem_sections(stem_points, num_sections=15)

            for section_points in section_list:
                z_coordinate = np.mean(section_points[:, 2])
                center, width, height, phi = utils.fit_ellipse(section_points)
                ellipse_points = utils.make_ellipse(center, width, height, phi, z_coordinate)

                # Post-process the ellipse to ensure it remains within the bounds of the stem
                stem_bbox = utils.get_bounding_box(stem_points)
                adjusted_ellipse_points = utils.post_process_ellipse(stem_bbox, ellipse_points, 
                                                                     center, phi, z_coordinate)

                #diameter = (width + height) / 2 TODO: calculate this
                #ellipse_diameters.append(diameter)

                visualization_ellipses.append(adjusted_ellipse_points)
        
            final_stems.append({"stem_points": stem_points, "ellipse_diameters": ellipse_diameters})

        self.final_stems = final_stems
        self.visualization_ellipses = visualization_ellipses