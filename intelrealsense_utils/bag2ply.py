import pyrealsense2 as rs
import os
import glob
root_dir = './IntelRealSense'
result_dir = './IntelRealSense_ply'

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

for bag_file in os.listdir(root_dir):
    bag_path = os.path.join(root_dir, bag_file)
    ply_fold = os.path.join(result_dir, bag_file[:-4])
    if not os.path.exists(ply_fold):
        os.mkdir(ply_fold)


    try:
        # Declare pointcloud object, for calculating pointclouods and texture mappings
        pc = rs.pointcloud()
        # We want the points object to be persistent so we can display the last cloud when a frame drops
        points = rs.points()

        # Declare RealSense pipeline, adapt a live-camera to a bag-reading script

        pipe = rs.pipeline()
        cfg = rs.config()
        rs.config.enable_device_from_file(cfg, bag_path, repeat_playback=False)

        # Start streaming with chonsen configuration
        profile = pipe.start(cfg)

        # Needed so frames don't get dropped during processing:
        profile.get_device().as_playback().set_real_time(False)

        frameNum = 0
        while True:

            # End loop once video finishes; Wait for the next set of frames from the camera
            frame_present, frames = pipe.try_wait_for_frames()
            if not frame_present:
                break

            # We will use the colorizer to generate texture for our PLY
            # (alternatively, texture can be obtained from color or infrared stream)
            colorizer = rs.colorizer()

            # Wait for the next set of frames from the camera
            # frames = pipe.wait_for_frames()
            # print(type(frames))

            colorized  = colorizer.process(frames)
            print('Processing ' + str(colorized.get_frame_number()) + ' frames in this .bag file.')

            # Create save_to_ply object

            ply = rs.save_to_ply(os.path.join(ply_fold, (str(frameNum) + ".ply")))


            # Set options to the desired values
            # In this example we will generate a textual PLY with normals (mesh is already created by default)
            ply.set_option(rs.save_to_ply.option_ply_binary, False)
            ply.set_option(rs.save_to_ply.option_ply_normals, True)

            print("Saving to" + str(frameNum) + ".ply...")
            # Apply the processing block to the frameset which contains the depth frame and the texture
            ply.process(colorized)
            print("Done")
            frameNum = frameNum + 1
    
    finally:
        # Stop streaming
        pipe.stop()

