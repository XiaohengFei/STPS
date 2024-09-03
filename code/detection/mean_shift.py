import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth

class StringDetectorMeanShift:
    def __init__(self) -> None:
        pass

    def detect_string(self, color_img: np.ndarray):
        cvt_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        cond = cvt_img[:, :, 0] < 10
        cond3 = np.dstack([cond] * 3)

        cond3 = cond3.reshape(-1, 3)
        IMG_H, IMG_W = color_img.shape[:2]
        uv_map = (
            np.array([[u, v, 1] for v in range(IMG_H) for u in range(IMG_W)])
            .reshape(-1, 3)
        )

        print(uv_map.shape)
        uv_map = uv_map[cond3].reshape(-1, 3)
        color_values = (cvt_img.reshape(-1, 3)[cond3]).reshape(-1, 3)[:, :2].reshape(-1, 2)
        X = np.hstack([color_values])
        print(f"X shape, ", X.shape)

        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
        print(bandwidth)

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        num_labels = len(np.unique(labels))
        print(num_labels)
        print(cluster_centers)

        seg_img = np.ones((IMG_H, IMG_W), np.int32) * num_labels
        print(seg_img.shape, seg_img[cond].shape, labels.shape)
        seg_img[cond] = labels
        seg_img = seg_img.reshape((IMG_H, IMG_W))
        seg_img = seg_img.astype(np.float32) / num_labels
        seg_img = np.dstack([seg_img] * 3)

        images = np.hstack([color_img / 255, seg_img])
        cv2.imshow("seg", images)
        cv2.waitKey(1)

        return None, None
