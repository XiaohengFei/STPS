import os
from src.detection.realsense_yolo import RealSenseYOLO
from src.analysis.gmm_analysis import GMMAnalyzer

def process_frame(frame, string_boxes):
    print("Processing frame...")
    # Initialising the GMM analyser
    gmm_analyzer = GMMAnalyzer()

    # Analysing cutting points
    cut_points = gmm_analyzer.find_cut_points(frame, string_boxes)

    # Returns the cut point for use in the main programme
    return cut_points

def main():
    print("Initializing YOLOv8 and RealSense...")
    model_path = os.path.abspath("E:\\Project\\STPS\\cutting\\models\\best.pt")
    detector = RealSenseYOLO(model_path=model_path)
    
    print("Starting detection...")
    detector.run(process_frame)

if __name__ == "__main__":
    print("Starting main program...")
    main()
