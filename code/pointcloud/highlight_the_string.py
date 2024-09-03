import time

import cv2
import numpy as np
import open3d as o3d

from utils.position_filter import PositionFilter


class StringHighligh:
    def __init__(self, show_3d: bool = True, show_fps: bool = True) -> None:
        """
        show_3d: if showing the 3d point cloud window
        """
        self._show_3d = show_3d
        self._show_fps = show_fps
        self._pos_mean_filter = PositionFilter(dim=3, pool_size=5)

        if show_3d:
            self._vis = o3d.visualization.Visualizer()
            self._camera_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.3, origin=[0, 0, 0]
            )
            self._pcd = o3d.geometry.PointCloud()

        if show_fps:
            self._last_time = 0
            self._counter = 0
            self._fps = 0

    def start(self, color_intrinsics: np.ndarray, depth_intrinsics: np.ndarray):
        """
        color_intrinsics: the intrinsics of color camera
        depth_intrinsics: the camera intrinsics of depth

        the structure of the intrinsics like below
        [width, height, Cx, Cy, Fx, Fy]
        """
        IMG_W = int(color_intrinsics[0])
        IMG_H = int(color_intrinsics[1])
        color_uv_map = (
            np.array([[u, v, 1] for v in range(IMG_H) for u in range(IMG_W)])
            .reshape(-1, 3)
            .T
        )

        color_Mp = np.array(
            [
                [color_intrinsics[4], 0, color_intrinsics[2]],
                [0, color_intrinsics[5], color_intrinsics[3]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        self._normalized_points = np.linalg.inv(color_Mp) @ color_uv_map

        # set the vis window
        if self._show_3d:
            self._vis.create_window(window_name="Bundle", width=640, height=480)
            self._vis.get_render_option().point_size = (
                self._vis.get_render_option().point_size * 0.2
            )
            self._vis.get_render_option().background_color = np.array([0, 0, 0])
            self._vis.add_geometry(self._camera_coord)

            self._first_frame = True

    def update(
        self,
        color_frame: np.ndarray,
        color_timestamp: float,
        depth_frame: np.ndarray,
        depth_timestamp: float,
        mask: np.ndarray,
    ):
        """
        color_frame: BGR-channel color image
        depth_frame: 16 bit gray scale image, unit in mm
        mask: the single channel mask to indicate the location of string
        """
        color_img = np.array(color_frame)

        # depth image is 1 channel, color is 3 channels
        depth_img = np.dstack([depth_frame] * 3)

        mask_3c = np.dstack((mask, mask, mask))

        # cv2.imshow("condition", cond.astype(np.int8).astype(np.float32))

        bg_removed = np.where(mask_3c, color_img, np.zeros_like(color_img))
        # bg_removed = cv2.circle(
        #     bg_removed, center, radius, color=(255, 255, 0), thickness=2
        # )

        images = np.hstack((color_img, bg_removed))

        depth_values = depth_frame * 1e-3
        # depth_values[depth_values > 2] = 0
        point3d = self._normalized_points * np.array(depth_values).reshape(1, -1)

        selected_point3d = point3d[:, mask.reshape(-1) == 1]
        if selected_point3d.shape[1] != 0:
            string_pos = np.median(selected_point3d, axis=1)
            string_pos = self._pos_mean_filter.update(string_pos)
        else:
            string_pos = [np.nan] * 3

        label_pos = (
            f"(X:{string_pos[0]:.2f}, Y:{string_pos[1]:.2f}, Z:{string_pos[2]:.2f})"
        )

        images = cv2.putText(
            images, label_pos, (color_img.shape[1], 30), 0, 1, (255, 255, 255), 2, 0
        )

        if self._show_3d:
            xyz = point3d.T
            # xyz[:, :2] = - xyz[:, :2]
            colors = np.where(mask_3c == 0, [0.4] * 3, [1, 1, 0]).reshape(-1, 3)

            point_filer = xyz[:, 2] < 2
            xyz = xyz[point_filer]
            colors = colors[point_filer]

            self._pcd.points = o3d.utility.Vector3dVector(xyz)
            self._pcd.colors = o3d.utility.Vector3dVector(colors)
            # pcd.normals = o3d.utility.Vector3dVector(np.array([[0, 0, 1]] * xyz.shape[0]))

            self._pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=5)
            )
            # pcd.paint_uniform_color([0.5, 0.5, 0.5])
            # pcd.orient_normals_to_align_with_direction()

            # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            # sphere = sphere.translate(string_pos)
            # pcd = (pcd, sphere)

            if self._first_frame:
                self._vis.add_geometry(self._pcd)
                self._first_frame = False
            else:
                self._vis.update_geometry(self._pcd)
            self._vis.poll_events()
            self._vis.update_renderer()

        if self._show_fps:
            self._counter += 1
            ts = time.time()
            time_diff = ts - self._last_time
            if time_diff > 1.0:
                self._fps = self._counter / time_diff
                self._counter = 0
                self._last_time = ts

            images = cv2.putText(
                images, f"FPS: {self._fps:.1f}", (10, 30), 0, 1, (255, 255, 255), 2, 0
            )

        return images

    def stop(self):
        if self._show_3d:
            self._vis.destroy_window()
