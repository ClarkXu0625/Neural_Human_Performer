import cv2
import os
import re

path = 'data/perform/thuman_nhp/epoch_-1/debug/0/'
vid_filename = 'thuman_subject_0'
os.makedirs('videos', exist_ok=True)

# Helper to sort by view number
def extract_view_index(filename):
    match = re.search(r'view(\d+)', filename)
    return int(match.group(1)) if match else -1

# Collect all view images for frame0
files = [f for f in os.listdir(path) if f.startswith('frame0_view') and f.endswith('.png')]
files.sort(key=extract_view_index)

# Set speed and initialize list
fps = 30
img_array = []

for file in files:
    img_path = os.path.join(path, file)
    img = cv2.imread(img_path)
    if img is None:
        continue
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

# Save to video
video_path = os.path.join('videos', vid_filename + '.mp4')
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for img in img_array:
    out.write(img)
out.release()

print("Saving spin video to:", os.path.abspath(video_path))
