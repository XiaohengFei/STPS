import os
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} sample_id")
    exit(1)

color_sample = []

try:
    data_dir = Path(os.environ["DATA_DIR"])
except KeyError as e:
    print(f"set environment variable '{e}'")
    exit()


sample_id = sys.argv[1]
count = len(list())
for fp in (Path(data_dir) / "ColorSamples" / f"{sample_id}").glob("*.*"):
    print(fp)
    file_path = str(fp)
    print(file_path)
    img = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_data = img.reshape((-1, 3))
    for sample in img_data:
        color_sample.append(sample)

print(f"sample size {len(color_sample)}")

color_sample = np.array(color_sample)
print(color_sample.shape)

mean = np.mean(color_sample, axis=0)
mean = mean.reshape((3,))
print(f"mean {mean}")
print(f"median {np.median(color_sample, axis=0)}")
mse = np.mean(np.square(color_sample - mean), axis=0)
print(f"MSE {np.mean(np.square(color_sample - mean), axis=0)}")

th_range = mse / 10
th_range[0] = 4

lower, higher = mean - th_range, mean + th_range
color_range = {"low": lower, "high": higher}

print(f"range {color_range}")

with open(str(Path(data_dir) / "color_range.pkl"), "wb") as f:
    pickle.dump(color_range, f)


blocks = 20
bs = 20
color_band = np.zeros((bs, bs * blocks, 3), dtype=np.uint8)
for i in range(blocks):
    alpha_a = float(i) / blocks
    alpha_b = 1 - alpha_a
    fill_color = lower * alpha_a + higher * alpha_b
    color_band[:, i * bs : (i + 1) * bs] = fill_color
    color_band[:, i * bs : (i + 1) * bs, 1:] = mean[1:]

color_band = cv2.cvtColor(color_band, cv2.COLOR_HSV2BGR)
cv2.imshow("color", color_band)
cv2.waitKey()
