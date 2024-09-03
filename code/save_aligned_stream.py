import pyrealsense2 as rs
import numpy as np
import cv2
import h5py
import time
import os
from pathlib import Path
import time

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
# clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Modify the dataset path to the desired directory
#data_path = r"C:\Users\Erin\OneDrive\桌面\Project\STPS\data\Bundles"

# Ensure the directory exists
#if not os.path.exists(data_path):
#    os.makedirs(data_path)

#file_name = "data_{:.3f}.hdf5".format(time.time())
#file_path = os.path.join(data_path, file_name)

dataset = h5py.File(f"dataset.{time.time():.3f}.hdf5", "w")

grab_intrinsics = True
frame_idx = 0
datastream = dataset.create_group("frames")
last_time = 0.0

START_SAVE = False

save_dir = Path(".") / f"{time.time():.2f}"
save_dir.mkdir()

# Streaming loop
idx = 0
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        if grab_intrinsics:
            def intrinsics(frame):
                intri = rs.video_stream_profile(frame.profile).get_intrinsics()
                return np.array([
                    intri.width, intri.height,
                    intri.ppx, intri.ppy,
                    intri.fx, intri.fy
                ], dtype=np.float32)

            # Grab intrinsics (may be changed by decimation)
            dataset.create_group("intrinsics")
            dataset["intrinsics"]["color"] = intrinsics(color_frame)
            dataset["intrinsics"]["depth"] = intrinsics(aligned_depth_frame)
            grab_intrinsics = False

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_timestamp = aligned_depth_frame.get_timestamp()
        color_image = np.asanyarray(color_frame.get_data())
        color_timestamp = color_frame.get_timestamp()

        cv2.imwrite(str(save_dir / f"color_{idx}.png"), color_image)
        cv2.imwrite(str(save_dir / f"depth_{idx}.png"), depth_image)
        idx += 1
        time.sleep(1)
        
        # Save NumPy arrays
        np.save(str(save_dir / f"color_{idx}.npy"), color_image)
        np.save(str(save_dir / f"depth_{idx}.npy"), depth_image)
        
        
        if START_SAVE:
            fid = f"{frame_idx}"
            datastream.create_group(fid)
            datastream[fid].create_group("depth")
            datastream[fid]["depth"].attrs["timestamp"] = f"{depth_timestamp:.3f}"
            datastream[fid]["depth"].create_dataset("frame", data=depth_image, compression='gzip')
            datastream[fid].create_group("color")
            datastream[fid]["color"].attrs["timestamp"] = f"{color_timestamp:.3f}"
            datastream[fid]["color"].create_dataset("frame", data=color_image, compression='gzip')
            frame_idx += 1
            print(f"cached frame {depth_timestamp}  {color_frame}")

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels

        # Render images:
        # depth align to color on left
        # depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if key & 0xFF == ord('s'):
            START_SAVE = True
finally:
    pipeline.stop()
    dataset.attrs["frame_count"] = frame_idx
    del dataset
