import os
from pathlib import Path
# import sys
from time import sleep

import cv2
import numpy as np
import pyrealsense2 as rs

# from detection.color_space_central import StringDetector
# from pointcloud.highlight_the_string import StringHighligh


# try:
#     color_range_file = Path(os.environ["DATA_DIR"]) / "color_range.pkl"
# except KeyError as e:
#     print(f"set environment variable '{e}'")
#     exit()

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)


config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

color_sensor = profile.get_device().first_color_sensor()
color_sensor.set_option(rs.option.enable_auto_exposure, False)
color_sensor.set_option(rs.option.exposure, 10)
print(color_sensor.get_option(rs.option.exposure))

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


grab_intrinsics = True

# radius = 40
# detector = StringDetector(color_range_file, radius=radius)
# ms_detector = StringDetectorMeanShift()

# SHOW_3D = False if len(sys.argv) == 1 else True
# string_highlight = StringHighligh(SHOW_3D)

cv2.namedWindow("live stream", cv2.WINDOW_NORMAL)
# Streaming loop

index = 0
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # if grab_intrinsics:

        #     def intrinsics(frame):
        #         intri = rs.video_stream_profile(frame.profile).get_intrinsics()
        #         return np.array(
        #             [
        #                 intri.width,
        #                 intri.height,
        #                 intri.ppx,
        #                 intri.ppy,
        #                 intri.fx,
        #                 intri.fy,
        #             ],
        #             dtype=np.float32,
        #         )

        #     # Grab intrinsics (may be changed by decimation)
        #     intrinsics_color = intrinsics(color_frame)
        #     intrinsics_depth = intrinsics(aligned_depth_frame)

        #     grab_intrinsics = False

        #     string_highlight.start(intrinsics_color, intrinsics_depth)

        depth_timestamp = aligned_depth_frame.get_timestamp()
        depth_frame = np.asanyarray(aligned_depth_frame.get_data())
        color_timestamp = color_frame.get_timestamp()
        color_frame = np.asanyarray(color_frame.get_data())


        # string_mask, _ = ms_detector.detect_string(color_frame)
        # color_frame[:, :200] = 0
        # string_mask, _ = detector.detect_string(color_frame)

        # images = string_highlight.update(
        #     color_frame, color_timestamp, depth_frame, depth_timestamp, string_mask
        # )

        images = color_frame
        cv2.imshow("live stream", images)
        key = cv2.waitKey(10)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 127:
            cv2.destroyAllWindows()
            break
        elif key &  0xFF == ord("s"):
            cv2.imwrite(f"pic/{index}_color_{color_timestamp:.6f}.png", color_frame)
            cv2.imwrite(f"pic/{index}_depth_{depth_timestamp:.6f}.png", depth_frame)

            index += 1

finally:
    pipeline.stop()
    # string_highlight.stop()
