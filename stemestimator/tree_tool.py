import os
import pclpy
import utils
import seg_tree
import numpy as np
import pandas as pd
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

    def set_point_cloud(self, point_cloud):
        """
        Resets the point cloud that treetool will process

        Args:
            point_cloud : np.narray | pclpy.pcl.PointCloud.PointXYZRGB | pclpy.pcl.PointCloud.PointXYZRGB
                The 3d point cloud of the forest that treetool will process, if it's a numpy array it should be shape (n,3)

        Returns:
            None
        """
        if point_cloud is not None:
            assert (
                (type(point_cloud) == pclpy.pcl.PointCloud.PointXYZRGB)
                or (type(point_cloud) == pclpy.pcl.PointCloud.PointXYZ)
                or (type(point_cloud) == np.ndarray)
            ), "Not valid point_cloud"
            if type(point_cloud) == np.ndarray:
                self.point_cloud = pclpy.pcl.PointCloud.PointXYZ(point_cloud)
            else:
                self.point_cloud = point_cloud

    def step_5_get_ground_level_trees(self, lowstems_height=5, cutstems_height=5, cut_stems=True):
        """Filters stems to only keep those near the ground and crops them up to a certain height.

        :param lowstems_height: The height threshold for low stems.
        :type lowstems_height: int
        :param cutstems_height: The height threshold for cutting stems.
        :type cutstems_height: int
        :param cut_stems: Whether to cut the stems.
        :type cut_stems: bool
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

        # Crop points above cutstems_height threshold
        if cut_stems:
            cut_stems = [
                [i[0][i[0][:, 2] < (cutstems_height + i[1][2])], i[1]] for i in low_stems
            ]
        else:
            cut_stems = low_stems

        self.cut_stems = cut_stems
        self.low_stems = [i[0] for i in cut_stems]

    def step_6_get_cylinder_tree_models(
        self, search_radius=0.1, distance=0.08, stick=False
    ):
        """
        For each cut stem we use ransac to extract a cylinder model

        Args:
            search_radius : float
                Maximum distance of the points to a sample point that will be used to approximate a the sample point's normal

        Returns:
            None
        """
        final_stems = []
        visualization_cylinders = []
        for p in self.cut_stems:
            # Segment to cylinders
            stem_points = p[0]
            if stick:
                indices, model = seg_tree.segment_normals(
                    stem_points,
                    search_radius=search_radius,
                    model=pclpy.pcl.sample_consensus.SACMODEL_STICK,
                    method=pclpy.pcl.sample_consensus.SAC_RANSAC,
                    normalweight=0.01,
                    miter=10000,
                    distance=0.4,
                    rlim=[0, 0.3],
                )
            else:
                indices, model = seg_tree.segment_normals(
                    stem_points,
                    search_radius=search_radius,
                    model=pclpy.pcl.sample_consensus.SACMODEL_CYLINDER,
                    method=pclpy.pcl.sample_consensus.SAC_RANSAC,
                    normalweight=0.01,
                    miter=10000,
                    distance=distance,
                    rlim=[0, 0.2],
                )
            # If the model has more than 10 points
            if len(indices) > 10:
                # If the model finds an upright cylinder
                if (
                    abs(np.dot(model[3:6], [0, 0, 1]) / np.linalg.norm(model[3:6]))
                    > 0.5
                ):
                    # Get centroid
                    model = np.array(model)
                    Z = 1.3 + p[1][2]
                    Y = model[1] + model[4] * (Z - model[2]) / model[5]
                    X = model[0] + model[3] * (Z - model[2]) / model[5]
                    model[0:3] = np.array([X, Y, Z])
                    # make sure the vector is pointing upward
                    model[3:6] = utils.similarize(model[3:6], [0, 0, 1])
                    final_stems.append({"tree": stem_points[indices], "model": model, 'ground': p[1][2]})
                    visualization_cylinders.append(
                        utils.makecylinder(model=model, height=7, density=60)
                    )

        self.finalstems = final_stems
        self.visualization_cylinders = visualization_cylinders

    def step_7_ellipse_fit(self, height_ll=-1,height_ul=-1):
        """
        Extract the cylinder and ellipse diameter of each stem

        Args:
            None

        Returns:
            None
        """
        for i in self.finalstems:
            # if the tree points has enough points to fit a ellipse
            if len(i["tree"]) > 5:
                # find a matrix that rotates the stem to be colinear to the z axis
                R = utils.rotation_matrix_from_vectors(i["model"][3:6], [0, 0, 1])
                # we center the stem to the origen then rotate it
                centeredtree = i["tree"] - i["model"][0:3]
                correctedcyl = (R @ centeredtree.T).T
                # fit an ellipse using only the xy coordinates
                try:
                    if height_ll != -1:
                        correctedcyl = correctedcyl[:,2]>height_ll
                    if height_ul != -1:
                        correctedcyl = correctedcyl[:,2]<height_ul
                    reg = LsqEllipse().fit(correctedcyl[:, 0:2])
                    center, a, b, phi = reg.as_parameters()

                    ellipse_diameter = 3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))
                except np.linalg.LinAlgError:
                    ellipse_diameter = i["model"][6] * 2
                except IndexError:
                    ellipse_diameter = i["model"][6] * 2
                cylinder_diameter = i["model"][6] * 2
                i["cylinder_diameter"] = cylinder_diameter
                i["ellipse_diameter"] = ellipse_diameter
                i["final_diameter"] = max(ellipse_diameter, cylinder_diameter)
                n_model = i["model"]
                n_model[6] = i["final_diameter"]
                i['vis_cyl'] = utils.makecylinder(model=n_model, height=7, density=60)
            else:
                i["cylinder_diameter"] = None
                i["ellipse_diameter"] = None
                i["final_diameter"] = None
                i['vis_cyl'] = None

    def save_results(self, save_location="results/myresults.csv"):
        """
        Save a csv with XYZ and DBH of each detected tree

        Args:
            savelocation : str
                path to save file

        Returns:
            None
        """
        tree_model_info = [i["model"] for i in self.finalstems]
        tree_diameter_info = [i["final_diameter"] for i in self.finalstems]

        data = {"X": [], "Y": [], "Z": [], "DBH": []}
        for i, j in zip(tree_model_info, tree_diameter_info):
            data["X"].append(i[0])
            data["Y"].append(i[1])
            data["Z"].append(i[2])
            data["DBH"].append(j)

        os.makedirs(os.path.dirname(save_location), exist_ok=True)

        pd.DataFrame.from_dict(data).to_csv(save_location)
