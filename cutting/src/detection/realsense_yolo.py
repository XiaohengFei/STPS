import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import os

class RealSenseYOLO:
    def __init__(self, model_path):
        print("Setting up RealSense pipeline...")
        # Initialising the RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # priming pipeline
        try:
            self.pipeline.start(self.config)
            print("RealSense pipeline started successfully.")
        except Exception as e:
            print(f"Error starting RealSense pipeline: {e}")

        # Load YOLOv8 Model
        try:
            model_path = os.path.abspath("E:\\Project\\STPS\\cutting\\models\\best.pt")
            print(f"Loading YOLOv8 model from: {model_path}")
            self.model = YOLO(model_path)
            print("YOLOv8 model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")

    def run(self, process_callback):
        try:
            while True:
                print("Waiting for frames...")
                # Waiting for a frame of data
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                # Confirmation that the frame has been captured correctly
                if not color_frame:
                    print("No frame captured, skipping...")
                    continue

                # Converting images to numpy arrays
                frame = np.asanyarray(color_frame.get_data())
                print("Frame captured.")

                if not hasattr(self, 'model'):
                    print("Model not loaded correctly, skipping frame processing...")
                    continue

                # Prediction using the YOLOv8 model
                results = self.model(frame)
                print("YOLOv8 detection completed.")

                # Extract the checkbox information for ‘string’.
                string_boxes = self.extract_string_boxes(results)
                print(f"Detected {len(string_boxes)} string boxes.")

                # Call the processing callback function for further processing
                cut_points = process_callback(frame, string_boxes)

                # Plotting cut points on the image
                self.draw_cut_points(frame, cut_points, string_boxes)

                # Plotting predictions on image
                annotated_frame = results[0].plot()

                # display frame
                cv2.imshow('YOLOv8 Real-Time Detection with RealSense', annotated_frame)

                # Press the ‘q’ key to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting on user command...")
                    break

        except Exception as e:
            print(f"An error occurred during frame processing: {e}")
        finally:
            # Stop the pipe flow
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def extract_string_boxes(self, results):
        # Extract the checkbox information for ‘string’
        string_boxes = []
        for result in results[0].boxes:
            # Verify that the category name and category ID match
            print(f"Detected class: {result.cls}, confidence: {result.conf}")

            if result.cls == 0:
                string_boxes.append(result.xyxy.numpy())

        print(f"Detected {len(string_boxes)} string boxes.")
        return string_boxes

    def draw_cut_points(self, frame, cut_points, string_boxes):
        for cut_point, box in zip(cut_points, string_boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.line(frame, (cut_point, y1), (cut_point, y2), (0, 0, 255), 2)
            print(f"Drawing cut point at: {cut_point} within box: [{x1}, {y1}, {x2}, {y2}]")

