import cv2
import numpy as np
import pickle

from utils.position_filter import PositionFilter


class StringDetector:
    F_WIN = 4

    def __init__(self, color_range_file, radius=25) -> None:
        self.frame_id = 0
        self.radius = radius
        self._center_filter = PositionFilter(dim=2, pool_size=StringDetector.F_WIN)

        self._color_range_file = color_range_file
        self._load_color_range(self._color_range_file)

    def _load_color_range(self, color_file_path):
        with open(color_file_path, "rb") as f:
            self.color_range = pickle.load(f)

    def detect_string(self, color_img):
        cvt_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        h_channel = cvt_img[:, :, 0]
        s_channel = cvt_img[:, :, 1]
        v_channel = cvt_img[:, :, 2]
        # cv2.imshow("h_channel", np.dstack([h_channel] * 3))
        # cv2.imshow("s_channel", np.dstack([s_channel] * 3))

        # print(cvt_img.shape)
        # print(color_range)

        h_low_cond = cvt_img[:, :, 0] > self.color_range["low"][0]
        h_high_cond = cvt_img[:, :, 0] < self.color_range["high"][0]

        s_low_cond = cvt_img[:, :, 1] > self.color_range["low"][1]
        s_high_cond = cvt_img[:, :, 1] < self.color_range["high"][1]

        v_low_cond = cvt_img[:, :, 2] > self.color_range["low"][2]
        v_high_cond = cvt_img[:, :, 2] < self.color_range["high"][2]

        cond = (
            h_low_cond
            & h_high_cond
            & v_low_cond
            & v_high_cond
            & s_low_cond
            & s_high_cond
        )

        kernel = np.ones((3, 2), np.uint8)
        cond = np.array(cond, np.uint8)
        cond = cv2.morphologyEx(cond, cv2.MORPH_OPEN, kernel)

        r_mean = np.argmax(np.sum(cond, axis=1))
        c_mean = np.argmax(np.sum(cond, axis=0))

        center = self._center_filter.update(np.array([c_mean, r_mean]))
        center = np.array(center).round().astype(np.int32).flatten()

        mask = np.zeros(color_img.shape[:2], dtype=np.uint8)
        mask = cv2.circle(mask, center, self.radius, 1, -1)

        mask = mask & cond

        # cv2.imshow("mask v", mask.astype(np.float32) * )

        self.frame_id += 1

        return mask, center
