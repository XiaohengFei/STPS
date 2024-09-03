import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import torch
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealSenseYOLO:
    def __init__(self, model_path):

        logger.info("Setting up RealSense pipeline...")

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        try:
            self.pipeline.start(self.config)
            logger.info("RealSense pipeline started successfully.")
        except Exception as e:
            logger.error("Error starting RealSense pipeline: %s", e)


        try:
            model_path = os.path.abspath(model_path)
            logger.info("Loading YOLOv8 model from: %s", model_path)
            self.model = YOLO(model_path)
            logger.info("YOLOv8 model loaded successfully.")
        except Exception as e:
            logger.error("Error loading YOLOv8 model: %s", e)

    def run(self, process_callback):

        try:
            while True:
                logger.info("Waiting for frames...")
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()


                if not color_frame or not depth_frame:
                    logger.warning("No frame captured, skipping...")
                    continue

                frame = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())

                depth_scaled = cv2.convertScaleAbs(depth, alpha=255.0 / np.max(depth))
                frame_4ch = cv2.merge((frame, depth_scaled))

                logger.info("Frame captured and depth channel merged.")

                if not hasattr(self, 'model'):
                    logger.warning("Model not loaded correctly, skipping frame processing...")
                    continue

                results = self.model(frame_4ch, conf=0.5)  
                logger.info("YOLOv8 detection completed.")

                string_boxes = self.extract_string_boxes(results)
                logger.info("Detected %d string boxes.", len(string_boxes))

                process_callback(frame, string_boxes)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Exiting on user command...")
                    break

        except Exception as e:
            logger.error("An error occurred during frame processing: %s", e)
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def extract_string_boxes(self, results):
        string_boxes = []
        for result in results[0].boxes:
            class_id = result.cls.item()
            confidence = result.conf.item()
            logger.info("Detected class: %f, confidence: %f", class_id, confidence)

            if class_id == 0:
                coords = result.xyxy.cpu().numpy().flatten()
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    string_boxes.append((x1, y1, x2, y2))

        return string_boxes

def process_frame(frame, string_boxes):
    logger.info("Processing frame...")
    gmm_analyzer = GMMAnalyzer(n_components=6)  

    combined_frame = frame.copy()

    for box in string_boxes:
        x1, y1, x2, y2 = map(int, box) 

        segment = frame[y1:y2, x1:x2]

        segmented_image = gmm_analyzer.segment_colors(segment)

        edges = gmm_analyzer.apply_canny_edge_detection(segment)

        cut_points = gmm_analyzer.find_cut_points(segment)

        colored_segment = gmm_analyzer.visualize_segmentation(segment, segmented_image, edges, cut_points)

        combined_frame[y1:y2, x1:x2] = colored_segment

        for idx, cut_point in enumerate(cut_points):
            cv2.putText(combined_frame, f"Cut Point {idx+1}: ({cut_point[0]}, {cut_point[1]})", (10, 50 + idx*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    result_image = cv2.hconcat([frame, combined_frame])

    cv2.imshow('Original and Segmented View with Cut Points', result_image)

def main():
    logger.info("Initializing YOLOv8 and RealSense...")
    model_path = os.path.abspath("yolov8_4channel1.pt")  
    detector = RealSenseYOLO(model_path=model_path)
    
    logger.info("Starting detection...")
    detector.run(process_frame)

if __name__ == "__main__":
    logger.info("Starting main program...")
    main()
