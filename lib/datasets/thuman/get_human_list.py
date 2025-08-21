import os
import re
from glob import glob
from sklearn.model_selection import train_test_split

def extract_human_ids(thuman_root):
    img_root = os.path.join(thuman_root, 'img')
    human_cams = os.listdir(img_root)

    human_ids = set()
    pattern = re.compile(r"^(\d{4})_\d{3}$")

    for folder in human_cams:
        match = pattern.match(folder)
        if match:
            human_ids.add(match.group(1))

    return sorted(human_ids)

def split_train_test(human_ids, test_ratio=0.2, seed=42):
    train_ids, test_ids = train_test_split(human_ids, test_size=test_ratio, random_state=seed)
    return sorted(train_ids), sorted(test_ids)

if __name__ == "__main__":
    thuman_data_root = 'data/THuman/val'  # Change if needed

    human_ids = extract_human_ids(thuman_data_root)
    print(f"Found {len(human_ids)} humans: {human_ids}")

    #train_ids, test_ids = split_train_test(human_ids)
    #print("\nTrain humans:", train_ids)
    #print("\nTest humans:", test_ids)
