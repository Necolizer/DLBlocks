import pyrealsense2 as rs
import os
import glob
import numpy as np
import cv2

root_dir = r'D:\ProgramData\IntelRealSense\20230409_170620.bag'
result_dir = r'D:\ProgramData\IntelRealSense'


bag_path = root_dir
ply_fold = os.path.join(result_dir, os.path.basename(root_dir)[:-4])
rgb_fold = os.path.join(result_dir, os.path.basename(root_dir)[:-4], 'rgb')
depth_fold = os.path.join(result_dir, os.path.basename(root_dir)[:-4], 'depth')

if not os.path.exists(ply_fold):
    os.makedirs(ply_fold)

if not os.path.exists(rgb_fold):
    os.makedirs(rgb_fold)

if not os.path.exists(depth_fold):
    os.makedirs(depth_fold)

try:

    # Declare pointcloud object, for calculating pointclouds and texture mappings
    pc = rs.pointcloud()
    # # We want the points object to be persistent so we can display the last cloud when a frame drops
    points = rs.points()

    # Declare RealSense pipeline, adapt a live-camera to a bag-reading script

    pipe = rs.pipeline()
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, bag_path, repeat_playback=False)

    # Start streaming with chonsen configuration
    profile = pipe.start(cfg)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Needed so frames don't get dropped during processing:
    profile.get_device().as_playback().set_real_time(False)

    frameNum = 0
    while True:

        # End loop once video finishes; Wait for the next set of frames from the camera
        frame_present, frames = pipe.try_wait_for_frames()
        if not frame_present:
            break

        if frameNum < 234:
            frameNum += 1
            continue

        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # depth_intrinsics = rs.video_stream_profile(
        #     depth_frame.profile).get_intrinsics()
        # depth_sensor = profile.get_device().first_depth_sensor()
        # depth_scale = depth_sensor.get_depth_scale()
        
        # print(depth_intrinsics)
        # print(depth_scale)
        
        # exit(1)

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(rgb_fold, str(frameNum) + '.jpg'), color_image)

        # # 获取16位深度图
        # depth_frame = np.asanyarray(aligned_depth_frame.get_data())
        # 利用中值核进行滤波
        depth_frame = rs.decimation_filter(1).process(depth_frame)
        # 从深度表示转换为视差表示，反之亦然
        depth_frame = rs.disparity_transform(True).process(depth_frame)
        # 空间滤镜通过使用alpha和delta设置计算帧来平滑图像。
        depth_frame = rs.spatial_filter().process(depth_frame)
        # 时间滤镜通过使用alpha和delta设置计算多个帧来平滑图像。
        depth_frame = rs.temporal_filter().process(depth_frame)
        # 从视差表示转换为深度表示
        depth_frame = rs.disparity_transform(False).process(depth_frame)
        # depth_frame = rs.hole_filling_filter().process(depth_frame)

        # 将深度图转化为RGB准备显示
        # depth_color_frame = rs.colorizer().colorize(depth_frame)
        # depth_color_image = np.asanyarray(depth_color_frame.get_data())

        depth_image = np.asanyarray(depth_frame.get_data())

        depths = depth_image
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

        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)

        o3d.io.write_point_cloud(r'D:\ProgramData\IntelRealSense\20230409_170620\temp.ply', cloud)

        exit(1)
        cv2.imwrite(os.path.join(depth_fold, 'depth_'+ str(frameNum) + '.jpg'), depth_image)


        # We will use the colorizer to generate texture for our PLY
        # (alternatively, texture can be obtained from color or infrared stream)
        colorizer = rs.colorizer()

        # Wait for the next set of frames from the camera
        # frames = pipe.wait_for_frames()
        # print(type(frames))

        # colorized  = colorizer.process(aligned_frames)
        # print('Processing ' + str(colorized.get_frame_number()) + ' frames in this .bag file.')

        # # Create save_to_ply object

        # ply = rs.save_to_ply(os.path.join(ply_fold, (str(frameNum) + ".ply")))


        # # Set options to the desired values
        # # In this example we will generate a textual PLY with normals (mesh is already created by default)
        # ply.set_option(rs.save_to_ply.option_ply_binary, False)
        # # ply.set_option(rs.save_to_ply.option_ply_normals, True)

        # print("Saving to" + str(frameNum) + ".ply...")
        # # Apply the processing block to the frameset which contains the depth frame and the texture
        # ply.process(colorized)
        # print("Done")
        frameNum = frameNum + 1
        # if frameNum == 410:
        #     exit(1)

finally:
    # Stop streaming
    pipe.stop()