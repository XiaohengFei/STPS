import cv2
import numpy as np
import os

# 定义图像文件夹路径
image_folder = '/networkhome/WMGDS/fei_x/workspace/1/dataset/dataset/dataset'  # 替换为实际路径
output_folder = '/networkhome/WMGDS/fei_x/workspace/1/dataset/dataset/uniondata'  # 替换为保存四通道图像的路径

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹内的所有文件
for filename in os.listdir(image_folder):
    if 'color' in filename:
        # 构造对应的深度图像文件名
        depth_filename = filename.replace('color', 'depth')
        
        # 构造完整的路径
        rgb_path = os.path.join(image_folder, filename)
        depth_path = os.path.join(image_folder, depth_filename)
        
        # 读取RGB和深度图像
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)  # 3-channel RGB
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)  # 1-channel Depth
        
        if rgb_img is not None and depth_img is not None:
            # 将深度图像扩展为与RGB图像相同的尺寸并将其添加为第四通道
            rgba_img = np.dstack((rgb_img, depth_img))
            
            # 保存合并后的四通道图像
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, rgba_img)
        else:
            print(f"Warning: Skipped {filename} or {depth_filename} due to loading error.")
