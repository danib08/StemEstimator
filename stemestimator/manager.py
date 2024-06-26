import utils
import pclpy
import seg_tree
import tree_tool
import numpy as np

class PointCloudManager:
    """Manages the point cloud file and its processing.

    :param file_path: The path to the point cloud file.
    :type file_path: str
    :param voxel_size: The size of the voxels to subsample the point cloud.
    :type voxel_size: float
    """
    def __init__(self, file_path, voxel_size=0.06):
        """Constructor method. Creates the pclpy point cloud.
        """
        self.point_cloud = self.create_point_cloud(file_path, voxel_size)
        self.tree_tool = tree_tool.TreeTool(self.point_cloud)

    def create_point_cloud(self, file_path, voxel_size=0.03):
        """Creates a point cloud from the input file and subsamples it.
        Converts the file if it is of .xyz type.

        :param file_path: The path to the input file.
        :type file_path: str
        :param voxel_size: The size of the voxels to subsample the point cloud.
        :type voxel_size: float
        :return: The point cloud as an array.
        :rtype: np.ndarray (n,3)
        """
        cloud = pclpy.pcl.PointCloud.PointXYZ()
        convert = file_path.endswith('.xyz')

        if convert:
            data = np.loadtxt(file_path, usecols=(2, 3, 4), dtype=np.float32)

            for point in data:
                pcl_point = pclpy.pcl.point_types.PointXYZ()
                pcl_point.x = point[0]
                pcl_point.y = point[1]
                pcl_point.z = point[2]
                cloud.push_back(pcl_point)
        else:
            pclpy.pcl.io.loadPCDFile(file_path, cloud)

        cloud_voxelized = seg_tree.voxelize(cloud.xyz, voxel_size, use_o3d=True)
        return cloud_voxelized

    def show_point_cloud(self):
        """Opens the point cloud in the Open3D viewer.

        :return: None
        """
        utils.open_3d_paint(self.point_cloud)
        
    def remove_ground(self, show=False):
        """Removes the floor from the point cloud.

        :param show: Whether to show the result in the Open3D viewer.
        :type show: bool
        :return: None
        """
        self.tree_tool.step_1_remove_ground()
        if show:
            utils.open_3d_paint([self.tree_tool.non_ground_cloud, self.tree_tool.ground_cloud])
            
    def normal_filtering(self, show=False):
        """Filters the point cloud based on normals.

        :param show: Whether to show the result in the Open3D viewer.
        :type show: bool
        :return: None
        """
        self.tree_tool.step_2_normal_filtering(search_radius=0.12, verticality_threshold=0.035, 
                                               curvature_threshold=0.04)
        if show:
            utils.open_3d_paint(self.tree_tool.filtered_points.xyz)

    def clustering(self, show=False):
        """Clusters the point cloud.

        :param show: Whether to show the result in the Open3D viewer.
        :type show: bool
        :return: None
        """
        self.tree_tool.step_3_euclidean_clustering(tolerance=0.4, min_cluster_size=40, max_cluster_size=500000)
        if show:
            utils.open_3d_paint(self.tree_tool.cluster_list)
    
    def group_stems(self, show=False):
        """Groups the clusters into stems.

        :param show: Whether to show the result in the Open3D viewer.
        :type show: bool
        :return: None
        """
        self.tree_tool.step_4_group_stems(max_distance=10)

        if show:
            utils.open_3d_paint(self.tree_tool.complete_stems, reduce_for_vis=False, voxel_size=0.1)

    def fit_ellipses(self):
        """Fits ellipses to the ground level trees.

        :return: None
        """
        self.tree_tool.step_5_fit_ellipses()

    def full_processing(self):
        """Runs the full processing of the point cloud.

        :return: None
        """
        self.remove_ground()
        self.normal_filtering()
        self.clustering()
        self.group_stems()
        self.fit_ellipses()
            
    def show_final_point_cloud(self):
        """Shows the final point cloud with the stems and ellipses with Open3D.

        :return: None
        """
        stem_data = self.tree_tool.final_stems
        utils.plot_full_cloud(stem_data)

    def get_tree_count(self):
        """Returns the number of trees in the point cloud.

        :return: int
        """
        return len(self.tree_tool.final_stems)
    
    def show_tree_results(self, tree_index):
        """Plot the selected tree and its ellipses using Plotly.

        :return: None
        """
        stem_data = self.tree_tool.final_stems[tree_index]
        utils.plot_tree(stem_data, tree_index)