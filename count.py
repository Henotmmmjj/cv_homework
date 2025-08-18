import os
from collections import Counter

label_dir = "C:\\yolov5-master\\dataset\\labels"  # 例如: 'data/labels/train'


# 手动定义类名映射（或也可从 classes.txt 读取）
class_names = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "-",
]

class_counts = Counter()

for file in os.listdir(label_dir):
    if file.endswith(".txt") and file != "classes.txt":
        file_path = os.path.join(label_dir, file)
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1

# 打印带标签名的统计结果
print("类别统计结果：\n")
for class_id in sorted(class_counts):
    label = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
    print(f"{label} (class {class_id}): {class_counts[class_id]} objects")
