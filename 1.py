import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from sklearn.mixture import GaussianMixture
import logging

# 设置日志记录
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GMMAnalyzer:
    def __init__(self, n_components=3):
        """初始化GMM分析器"""
        print("Initializing GMM analyzer...")
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full')

    def segment_colors(self, segment):
        """使用GMM分割颜色区域"""
        # 转换为Lab颜色空间以更好地分割颜色
        lab_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2Lab)
        
        # 将图像转换为2D数据点（每个像素三个通道）
        pixels = lab_segment.reshape(-1, 3)

        # 使用GMM模型进行拟合
        self.gmm.fit(pixels)

        # 预测每个像素的类别
        labels = self.gmm.predict(pixels)
        
        # 通过标签将图像重构为各个颜色区域
        segmented_image = labels.reshape(segment.shape[0], segment.shape[1])
        
        return segmented_image

    def visualize_segmentation(self, segment, segmented_image):
        """可视化分割结果"""
        # 创建一个随机颜色表，用于可视化不同的类别
        unique_labels = np.unique(segmented_image)
        colors = np.random.randint(0, 255, size=(len(unique_labels), 3))
        
        # 创建一个空的图像用于可视化
        colored_segment = np.zeros_like(segment)
        
        for label, color in zip(unique_labels, colors):
            colored_segment[segmented_image == label] = color
        
        # 显示分割结果
        cv2.imshow('Segmented Colors', colored_segment)
        cv2.waitKey(1)

class RealSenseYOLO:
    def __init__(self, model_path):
        """初始化RealSense管道和YOLOv8模型"""
        print("Setting up RealSense pipeline...")
        # 初始化RealSense管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # 启动管道
        try:
            self.pipeline.start(self.config)
            print("RealSense pipeline started successfully.")
        except Exception as e:
            print(f"Error starting RealSense pipeline: {e}")

        # 加载YOLOv8模型
        try:
            model_path = os.path.abspath(model_path)
            print(f"Loading YOLOv8 model from: {model_path}")
            self.model = YOLO(model_path)
            print("YOLOv8 model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")

    def run(self, process_callback):
        """实时检测并处理每一帧"""
        try:
            while True:
                print("Waiting for frames...")
                # 等待一帧数据
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                # 确认帧已被正确捕获
                if not color_frame:
                    print("No frame captured, skipping...")
                    continue

                # 将图像转换为numpy数组
                frame = np.asanyarray(color_frame.get_data())
                print("Frame captured.")

                if not hasattr(self, 'model'):
                    print("Model not loaded correctly, skipping frame processing...")
                    continue

                # 使用YOLOv8模型进行预测
                results = self.model(frame, conf=0.5)  # 设置置信度阈值为50%
                print("YOLOv8 detection completed.")

                # 提取“string”的检测框信息
                string_boxes = self.extract_string_boxes(results)
                print(f"Detected {len(string_boxes)} string boxes.")

                # 调用处理回调函数进行进一步处理
                process_callback(frame, string_boxes)

                # 在图像上绘制预测结果
                annotated_frame = results[0].plot()

                # 显示帧
                cv2.imshow('YOLOv8 Real-Time Detection with RealSense', annotated_frame)

                # 按下 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting on user command...")
                    break

        except Exception as e:
            print(f"An error occurred during frame processing: {e}")
        finally:
            # 停止管道流
            self.pipeline.stop()
            # 关闭窗口
            cv2.destroyAllWindows()

    def extract_string_boxes(self, results):
        """提取‘string’类别的检测框信息"""
        string_boxes = []
        for result in results[0].boxes:
            # 使用 .item() 方法将 tensor 转换为标量
            class_id = result.cls.item()
            confidence = result.conf.item()
            print(f"Detected class: {class_id}, confidence: {confidence}")

            # 假设类别 'string' 的ID是0
            if class_id == 0:
                # 将边界框坐标转换为标量
                coords = result.xyxy.cpu().numpy().flatten()
                if len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    string_boxes.append((x1, y1, x2, y2))

        print(f"Detected {len(string_boxes)} string boxes.")
        return string_boxes

def process_frame(frame, string_boxes):
    """处理每一帧图像，进行颜色分割"""
    print("Processing frame...")
    # 初始化GMM分析器
    gmm_analyzer = GMMAnalyzer()

    for box in string_boxes:
        x1, y1, x2, y2 = map(int, box)  # 将坐标转换为整数

        # 裁剪检测到的区域
        segment = frame[y1:y2, x1:x2]

        # 进行颜色分割
        segmented_image = gmm_analyzer.segment_colors(segment)

        # 可视化分割结果
        gmm_analyzer.visualize_segmentation(segment, segmented_image)

def main():
    """主函数，初始化检测器并开始检测"""
    print("Initializing YOLOv8 and RealSense...")
    model_path = os.path.abspath("E:\\Project\\STPS\\best.pt")
    detector = RealSenseYOLO(model_path=model_path)
    
    print("Starting detection...")
    detector.run(process_frame)

if __name__ == "__main__":
    print("Starting main program...")
    main()
