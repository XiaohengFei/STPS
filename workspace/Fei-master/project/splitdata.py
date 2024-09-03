import os
import random
import shutil

def get_file_triplets(data_dir, label_dir):
    """获取RGB图像、深度图像和标签文件的配对列表"""
    all_files = os.listdir(data_dir)
    label_files = os.listdir(label_dir)
    
    # 提取RGB和深度图像文件
    rgb_files = [f for f in all_files if '_color_' in f and f.endswith('.png')]
    depth_files = [f for f in all_files if '_depth_' in f and f.endswith('.png')]

    # 打印文件列表
    print(f"RGB files: {rgb_files}")
    print(f"Depth files: {depth_files}")
    print(f"Label files: {label_files}")

    triplet_files = []
    for rgb_file in rgb_files:
        base_name = rgb_file.replace('_color_', '_depth_')
        # 标签文件与RGB文件完全相同
        label_file = rgb_file.rsplit('.', 1)[0] + '.txt'

        # 打印调试信息
        print(f"Checking for RGB={rgb_file}, Depth={base_name}, Label={label_file}")

        if base_name in depth_files and label_file in label_files:
            triplet_files.append((rgb_file, base_name, label_file))
        else:
            # 打印调试信息，查看哪些文件没有匹配上
            if base_name not in depth_files:
                print(f"Missing Depth file: {base_name}")
            if label_file not in label_files:
                print(f"Missing Label file: {label_file}")
            print(f"Missing match: RGB={rgb_file}, Depth={base_name}, Label={label_file}")
    
    return triplet_files

# 原数据集目录
data_dir = 'project/data/dataset'
label_dir = 'project/data/labels'

# 调试输出，检查路径是否正确
print("Data directory:", data_dir)
print("Label directory:", label_dir)

# 检查路径是否存在
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory not found: {data_dir}")
if not os.path.exists(label_dir):
    raise FileNotFoundError(f"Label directory not found: {label_dir}")

# 拆分后数据集目录
split_dir = 'project/data/split'
os.makedirs(os.path.join(split_dir, 'train/rgb'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'train/depth'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'valid/rgb'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'valid/depth'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'valid/labels'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'test/rgb'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'test/depth'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'test/labels'), exist_ok=True)

# 获取RGB图像、深度图像和标签文件的配对列表
triplet_files = get_file_triplets(data_dir, label_dir)

# 打印配对文件的数量
print(f"Total paired files: {len(triplet_files)}")

# 随机打乱文件列表
random.seed(42)
random.shuffle(triplet_files)

# 根据比例计算划分的边界索引
total_files = len(triplet_files)
train_bound = int(total_files * 0.8)
valid_bound = int(total_files * 0.9)

# 将文件移动到相应的目录
for i, (rgb_file, depth_file, label_file) in enumerate(triplet_files):
    if i < train_bound:
        shutil.copy(os.path.join(data_dir, rgb_file), os.path.join(split_dir, 'train/rgb', rgb_file))
        shutil.copy(os.path.join(data_dir, depth_file), os.path.join(split_dir, 'train/depth', depth_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(split_dir, 'train/labels', label_file))
    elif i < valid_bound:
        shutil.copy(os.path.join(data_dir, rgb_file), os.path.join(split_dir, 'valid/rgb', rgb_file))
        shutil.copy(os.path.join(data_dir, depth_file), os.path.join(split_dir, 'valid/depth', depth_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(split_dir, 'valid/labels', label_file))
    else:
        shutil.copy(os.path.join(data_dir, rgb_file), os.path.join(split_dir, 'test/rgb', rgb_file))
        shutil.copy(os.path.join(data_dir, depth_file), os.path.join(split_dir, 'test/depth', depth_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(split_dir, 'test/labels', label_file))

print("数据集划分完成")
