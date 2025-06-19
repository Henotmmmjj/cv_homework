import albumentations as A
import cv2
import os
from tqdm import tqdm

# 配置路径
img_dir = 'dataset/images/'
label_dir = 'dataset/labels/'
save_img_dir = 'dataset/images_aug/'
save_label_dir = 'dataset/labels_aug/'

os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_label_dir, exist_ok=True)

# 定义增强 pipeline
transform = A.Compose([
    A.GaussNoise(var_limit=(10, 30), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 3), p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 增强次数（每张原图增强出 N 张）
augment_times = 1

for img_name in tqdm(os.listdir(img_dir)):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + '.txt')

    # 读取图像和标签
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    if not os.path.exists(label_path):
        continue

    with open(label_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, bw, bh = map(float, parts)
        bboxes.append([x, y, bw, bh])
        class_labels.append(int(cls))

    for i in range(augment_times):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        # 保存增强图像和对应标签
        aug_name = img_name.rsplit('.', 1)[0] + f'_aug{i}.jpg'
        out_img_path = os.path.join(save_img_dir, aug_name)
        out_label_path = os.path.join(save_label_dir, aug_name.replace('.jpg', '.txt'))

        cv2.imwrite(out_img_path, aug_img)

        with open(out_label_path, 'w') as f:
            for cls, box in zip(aug_labels, aug_bboxes):
                f.write(f"{cls} {' '.join(f'{b:.6f}' for b in box)}\n")
