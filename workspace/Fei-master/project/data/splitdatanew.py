import os
import random
import shutil

def get_file_pairs(image_dir, label_dir):
    """获取RGB图像和标签文件的配对列表"""
    image_files = os.listdir(image_dir)
    label_files = os.listdir(label_dir)
    
    # 提取以png结尾的图像文件
    image_files = [f for f in image_files if f.endswith(( '.png'))]
    
    pairs = []
    for image_file in image_files:
        base_name = image_file.rsplit('.', 1)[0]
        label_file = f"{base_name}.txt"
        if label_file in label_files:
            pairs.append((image_file, label_file))
        else:
            # 打印没有找到匹配的标签文件，用于调试
            print(f"No label for image: {image_file}")

    return pairs

# 原数据集目录
image_dir = 'project/data/uniondata'
label_dir = 'project/data/labelsnew'

# 调试输出，检查路径是否正确
print("Image directory:", image_dir)
print("Label directory:", label_dir)

# 检查路径是否存在
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Image directory not found: {image_dir}")
if not os.path.exists(label_dir):
    raise FileNotFoundError(f"Label directory not found: {label_dir}")

# 拆分后数据集目录
split_dir = 'project/data/split_new4'
os.makedirs(os.path.join(split_dir, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'valid/images'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'valid/labels'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'test/images'), exist_ok=True)
os.makedirs(os.path.join(split_dir, 'test/labels'), exist_ok=True)

# 获取RGB图像和标签文件的配对列表
image_label_pairs = get_file_pairs(image_dir, label_dir)

# 打印配对文件的数量
print(f"Total paired files: {len(image_label_pairs)}")

# 随机打乱文件列表
random.seed(42)
random.shuffle(image_label_pairs)

# 根据比例计算划分的边界索引
total_pairs = len(image_label_pairs)
train_bound = int(total_pairs * 0.8)
valid_bound = int(total_pairs * 0.9)

# 将文件复制到相应的目录
for i, (image_file, label_file) in enumerate(image_label_pairs):
    if i < train_bound:
        shutil.copy(os.path.join(image_dir, image_file), os.path.join(split_dir, 'train/images', image_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(split_dir, 'train/labels', label_file))
    elif i < valid_bound:
        shutil.copy(os.path.join(image_dir, image_file), os.path.join(split_dir, 'valid/images', image_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(split_dir, 'valid/labels', label_file))
    else:
        shutil.copy(os.path.join(image_dir, image_file), os.path.join(split_dir, 'test/images', image_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(split_dir, 'test/labels', label_file))

print("数据集划分完成")
