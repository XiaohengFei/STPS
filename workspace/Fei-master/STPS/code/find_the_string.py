from sys import argv

import cv2
import numpy as np

from detection.color_space_central import StringDetector
import h5py
from pointcloud.highlight_the_string import StringHighligh
from utils.frame_control import FrameControl

if len(argv) != 2:
    print(f"Usage: {argv[0]} file.hdf5")
    exit()


file_path = argv[1]
dataset = h5py.File(file_path, "r")

frame_count = dataset.attrs["frame_count"]
print(f"dataset: frame_count {frame_count}")

intrinsics_color = dataset["intrinsics"]["color"][:]
intrinsics_depth = dataset["intrinsics"]["depth"][:]
print(f"color intrinsics {intrinsics_color}")
print(f"depth intrinsics {intrinsics_depth}")


frames = dataset["frames"]

radius = 30
detector = StringDetector(radius=radius)


SHOW_3D = True
string_highlight = StringHighligh()
frame_control = FrameControl(frequency=10)

string_highlight.start(intrinsics_color, intrinsics_depth)

try:
    for i in range(frame_count):
        frmae = frames[f"{i}"]
        depth_timestamp = frmae["depth"].attrs["timestamp"]
        depth_frame = frmae["depth"]["frame"][:]
        mean_v = np.mean(depth_frame)
        # print(
        #     f"depth:\
        #     time {depth_timestamp}\
        #     frame shape {depth_frame.shape}\
        #     mean value {mean_v}"
        # )

        color_timestamp = frmae["color"].attrs["timestamp"]
        color_frame = frmae["color"]["frame"][:]
        mean_v = np.mean(color_frame)
        # print(
        #     f"color:\
        #     time {color_timestamp}\
        #     frame shape {color_frame.shape}\
        #     mean value {mean_v}"
        # )
        # print()

        if not frame_control.check(float(color_timestamp)):
            continue

        string_mask, _ = detector.detect_string(color_frame)

        images = string_highlight.update(
            color_frame, color_timestamp, depth_frame, depth_timestamp, string_mask
        )

        cv2.imshow("read from stream file", images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 127:
            cv2.destroyAllWindows()
            break

finally:
    dataset.close()
    string_highlight.stop()
