import os
import shutil
import random

# 设置随机种子以确保可重复性
random.seed(42)

# 定义文件夹路径
merged_images_dir = '/networkhome/WMGDS/fei_x/workspace/1/dataset/dataset/uniondata'  # 合并后图像的目录
labels_dir = '1/dataset/dataset/labels'  # 标签文件的目录
output_dir = '/networkhome/WMGDS/fei_x/workspace/1/dataset/dataset/string4'  # 输出数据集的目录

# 创建数据集目录结构
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

# 获取所有图像文件名
images = [f for f in os.listdir(merged_images_dir) if f.endswith('.png')]  # 根据实际情况选择图像格式

# 随机打乱图像列表
random.shuffle(images)

# 定义数据集划分比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 计算每个集合的样本数量
train_count = int(train_ratio * len(images))
val_count = int(val_ratio * len(images))

# 划分数据集
train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]

# 将图像和对应的标签移动到相应的目录
def move_files(images_list, split):
    for image_name in images_list:
        label_name = image_name.replace('.png', '.txt')  # 根据标签文件的实际命名规则替换扩展名
        
        # 检查标签文件是否存在
        label_path = os.path.join(labels_dir, label_name)
        if not os.path.exists(label_path):
            # 如果标签文件不存在，则删除该图像文件并跳过
            print(f"Warning: Skipping {image_name} due to missing label file.")
            image_path = os.path.join(merged_images_dir, image_name)
            os.remove(image_path)
            continue

        # 移动图像文件
        shutil.copy(os.path.join(merged_images_dir, image_name), 
                    os.path.join(output_dir, split, 'images', image_name))
        
        # 移动标签文件
        shutil.copy(label_path, 
                    os.path.join(output_dir, split, 'labels', label_name))

move_files(train_images, 'train')
move_files(val_images, 'val')
move_files(test_images, 'test')

print("数据集划分完成！")
