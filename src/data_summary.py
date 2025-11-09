import os
from collections import Counter

DATA_DIR = "data"

def count_images(base_dir):
    summary = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(base_dir, split)
        class_counts = {}
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[class_name] = count
        summary[split] = class_counts
    return summary

summary = count_images(DATA_DIR)

# Display neatly
for split, counts in summary.items():
    print(f"\n {split.upper()} SET:")
    total = sum(counts.values())
    for cls, cnt in counts.items():
        print(f"  {cls:<20} : {cnt}")
    print(f"  ➤ Total images: {total}")
