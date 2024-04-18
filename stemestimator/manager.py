import pclpy
import numpy as np
import open3d as o3d
import treetool.utils as utils
import treetool.seg_tree as seg_tree
import treetool.tree_tool as tree_tool

class PointCloudManager:
    def __init__(self, file_path):
        self.point_cloud_v = self.create_point_cloud(file_path)

    def create_point_cloud(self, file_path):
        convert = file_path.endswith('.xyz')
        new_filepath = file_path
        if convert:
            data = np.loadtxt(file_path, usecols=(2, 3, 4, 5, 6, 7), dtype=np.float32)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data[:, :3]) # x, y, z columns
            #pcd.colors = o3d.utility.Vector3dVector(data[:, 3:]) # r, g, b columns

            # save the point cloud as PCD
            new_filepath = './assets/plantacion_melina2.pcd'
            o3d.io.write_point_cloud(new_filepath, pcd)

        point_cloud = pclpy.pcl.PointCloud.PointXYZ()
        pclpy.pcl.io.loadPCDFile(new_filepath, point_cloud)
   
        self.point_cloud_v = seg_tree.voxelize(point_cloud.xyz, leaf=0.06, use_o3d=True)
        self.tree_tool = tree_tool.treetool(self.point_cloud_v)

    def show_point_cloud(self):
        utils.open3dpaint(self.point_cloud_v, reduce_for_vis=False, voxel_size=0.1)
        
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