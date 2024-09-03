import h5py
import numpy as np
from sys import argv
import cv2

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

for i in range(frame_count):
    frmae = frames[f"{i}"]
    depth_timestamp = frmae["depth"].attrs["timestamp"]
    depth_frame = frmae["depth"]["frame"][:]
    mean_v = np.mean(depth_frame)
    print(f"depth:\
        time {depth_timestamp}\
        frame shape {depth_frame.shape}\
        mean value {mean_v}")

    color_timestamp = frmae["color"].attrs["timestamp"]
    color_frame = frmae["color"]["frame"][:]
    mean_v = np.mean(color_frame)
    print(f"color:\
        time {color_timestamp}\
        frame shape {color_frame.shape}\
        mean value {mean_v}")
    print()

    color_img = np.array(color_frame)

    # depth image is 1 channel, color is 3 channels
    depth_img = np.dstack((depth_frame, depth_frame, depth_frame))

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((color_img, depth_colormap))
    cv2.imshow("read from stream file", images)

    key = cv2.waitKey(50)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break