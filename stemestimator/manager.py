import utils
import pclpy
import seg_tree
import tree_tool
import numpy as np


class PointCloudManager:
    """Manages the point cloud file and its processing.

    :param file_path: The path to the point cloud file.
    :type file_path: str
    """
    def __init__(self, file_path):
        """Constructor method. Creates the pclpy point cloud.
        """
        self.point_cloud = self.create_point_cloud(file_path)
        self.tree_tool = tree_tool.treetool(self.point_cloud) #TODO: check for significative name

    def create_point_cloud(self, file_path):
        """Creates a point cloud from the input file.
        Converts the file if it is of .xyz type.

        :param file_path: The path to the input file.
        :type file_path: str
        :return: The point cloud.
        :rtype: pclpy.pcl.PointCloud.PointXYZ
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

        cloud_voxelized = seg_tree.voxelize(cloud.xyz, leaf=0.06, use_o3d=True)
        return cloud_voxelized

    def show_point_cloud(self):
        """Opens the point cloud in the Open3D viewer.
        """
        utils.open3dpaint(self.point_cloud, reduce_for_vis=False)
        
    """
    def remove_floor(self, show=False):
        self.tree_tool.step_1_remove_floor()
        if show:
            utils.open3dpaint([self.tree_tool.non_ground_cloud, self.tree_tool.ground_cloud], 
                          reduce_for_vis=True, voxel_size=0.1)
        
    def normal_filtering(self, show=False):
        self.tree_tool.step_2_normal_filtering(verticality_threshold=0.04,
                                               curvature_threshold=0.06, search_radius=0.12)
        if show:
            utils.open3dpaint([self.tree_tool.non_ground_cloud.xyz, self.tree_tool.non_filtered_points.xyz + 
                            self.tree_tool.non_filtered_normals * 0.1, self.tree_tool.non_filtered_points.xyz +
                            self.tree_tool.non_filtered_normals * 0.2], reduce_for_vis=True, voxel_size=0.1)
            
            utils.open3dpaint([self.tree_tool.filtered_points.xyz, self.tree_tool.filtered_points.xyz +
                            self.tree_tool.filtered_normals * 0.05, self.tree_tool.filtered_points.xyz +
                            self.tree_tool.filtered_normals * 0.1], reduce_for_vis=True, voxel_size=0.1)

    def clustering(self, show=False):
        self.tree_tool.step_3_euclidean_clustering(tolerance=0.2, min_cluster_size=40, max_cluster_size=6000000)
        if show:
            utils.open3dpaint(self.tree_tool.cluster_list, reduce_for_vis=True, voxel_size=0.1)

    def group_stems(self, show=False):
        self.tree_tool.step_4_group_stems(max_distance=0.4)

        if show:
            utils.open3dpaint(self.tree_tool.complete_Stems, reduce_for_vis=True, voxel_size=0.1)

    def get_ground_level_trees(self, show=False):
        self.tree_tool.step_5_get_ground_level_trees(lowstems_height=5, cutstems_height=5)

        if show:
            utils.open3dpaint(self.tree_tool.low_stems, reduce_for_vis=True, voxel_size=0.1)

    def get_cylinders(self, show=False):
        self.tree_tool.step_6_get_cylinder_tree_models(search_radius=0.1)

        if show:
            utils.open3dpaint([i['tree'] for i in self.tree_tool.finalstems] +
                              self.tree_tool.visualization_cylinders, reduce_for_vis=True, voxel_size=0.1)
            
    def ellipse_fit(self):
        self.tree_tool.step_7_ellipse_fit()

    def save_results(self):
        self.tree_tool.save_results(save_location='results/myresults.csv')
    """