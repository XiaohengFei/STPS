from sklearn.mixture import GaussianMixture
import numpy as np
import cv2

class GMMAnalyzer:
    def __init__(self, n_components=3):  # 可以调整组件数量
        print("Initializing GMM analyzer...")
        self.gmm = GaussianMixture(n_components=n_components)

    def find_cut_points(self, frame, string_boxes):
        cut_points = []
        print("Analyzing cut points...")

        for box in string_boxes:
            x1, y1, x2, y2 = map(int, box)

            # Cropping the rope part of the image
            rope_segment = frame[y1:y2, x1:x2]

            # Convert to greyscale
            gray_segment = cv2.cvtColor(rope_segment, cv2.COLOR_BGR2GRAY)

            # Convert images to 2D data points
            pixels = gray_segment.reshape(-1, 1)

            # Fitting using a GMM model
            self.gmm.fit(pixels)

            # Find the mean of each Gaussian distribution
            means = self.gmm.means_.flatten()

            # Find the smallest Gaussian mean as the cut point
            cut_point_index = np.argmin(means)
            cut_point_value = means[cut_point_index]

            # Prints out all Gaussian distribution means for debugging purposes
            print(f"GMM Means: {means}")
            print(f"Chosen cut point mean: {cut_point_value}")

            # Ensure that scalar values are used for position calculations
            cut_point_position = int(cut_point_value) + x1
            cut_points.append(cut_point_position)

            # Visualisation of cutting points
            self.visualize_cut_point(rope_segment, cut_point_position, x1, y1, x2, y2)

        return cut_points

    def visualize_cut_point(self, segment, cut_point, x1, y1, x2, y2):
        # Marking cut points in an image
        height, _ = segment.shape[:2]
        cv2.line(segment, (cut_point - x1, 0), (cut_point - x1, height), (0, 255, 0), 2)

        # Output cutting point information
        print(f"Cutting point position：{cut_point}, Rope coordinates: [{x1}, {y1}, {x2}, {y2}]")

        # Show cut point visualisation
        cv2.imshow('Cut Point', segment)
        cv2.waitKey(1)  # Use 1 to ensure cut-point display in continuous video streams
