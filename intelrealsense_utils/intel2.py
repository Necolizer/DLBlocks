import numpy as np
from PIL import Image
import os
import open3d as o3d

# get data
colors = np.array(Image.open(r'D:\ProgramData\IntelRealSense\20230409_170620\rgb\204.jpg'), dtype=np.float32) / 255.0
depths = np.array(Image.open(r'D:\ProgramData\IntelRealSense\20230409_170620\depth\depth_204.jpg'))

# get camera intrinsics
fx, fy = 894.399, 894.749
cx, cy = 650.197, 358.816
scale = 0.00025

# set workspace
# xmin, xmax = -0.19, 0.12
# ymin, ymax = 0.02, 0.15
# zmin, zmax = 0.0, 1.0
# lims = [xmin, xmax, ymin, ymax, zmin, zmax]

# get point cloud
xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
xmap, ymap = np.meshgrid(xmap, ymap)
points_z = depths / scale
points_x = (xmap - cx) / fx * points_z
points_y = (ymap - cy) / fy * points_z

points = np.stack([points_x, points_y, points_z], axis=-1)
points = points.astype(np.float32)

points = points.reshape(-1, 3)
print(type(points))
print(points.shape)

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)

o3d.io.write_point_cloud(r'D:\ProgramData\IntelRealSense\20230409_170620\temp.ply', cloud)
