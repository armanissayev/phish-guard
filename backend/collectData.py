import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("taruntiwarihp/phishing-site-urls")

# Define your target directory (current directory or another, e.g., 'data/csv')
target_dir = os.path.join(os.getcwd(), "data", "csv")
os.makedirs(target_dir, exist_ok=True)

# Move all files from the downloaded path to the target directory
for filename in os.listdir(path):
    src = os.path.join(path, filename)
    dst = os.path.join(target_dir, filename)
    if os.path.isdir(src):
        shutil.move(src, dst)
    else:
        shutil.move(src, dst)

print("Dataset moved to:", target_dir)