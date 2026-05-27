"""
download_data.py
----------------
Downloads NIH ChestX-ray14 dataset from Kaggle to Google Drive.
Run once from Colab — never needs to run again.

Usage in Colab:
    from src.download_data import download_nih, verify_data
    download_nih(drive_root='/content/drive/MyDrive/data/nih')
"""

import os
import subprocess


def download_nih(drive_root: str = '/content/drive/MyDrive/data/nih',
                 extract_dir: str = '/content/extracted'):
    """
    Download and extract full NIH ChestX-ray14 dataset to Google Drive.

    Steps:
        1. Download full data.zip from Kaggle (~42GB) to /content
        2. Extract zip to /content/extracted
        3. Move all images and CSVs to Google Drive permanently

    Args:
        drive_root:   destination folder in Google Drive
        extract_dir:  temp folder for extraction (in Colab, not Drive)
    """
    os.makedirs(drive_root, exist_ok=True)
    img_dir = os.path.join(drive_root, 'images')
    os.makedirs(img_dir, exist_ok=True)

    # Step 1: Download full dataset
    zip_path = '/content/data.zip'
    if os.path.exists(zip_path):
        print('data.zip already downloaded, skipping.')
    else:
        print('Downloading full NIH dataset (~42GB)...')
        print('Takes ~5 minutes on Colab Pro.')
        result = subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', 'nih-chest-xrays/data',
            '--path', '/content',
            '--force'
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print('Download failed! Error:', result.stderr[:300])
            return
        print('Download complete!')

    # Step 2: Extract zip
    if os.path.exists(extract_dir):
        print('Already extracted, skipping.')
    else:
        print('Extracting data.zip (10-15 mins)...')
        result = subprocess.run([
            'unzip', '-q', zip_path, '-d', extract_dir
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print('Extraction failed! Error:', result.stderr[:300])
            return
        print('Extraction complete!')

    # Step 3: Move CSVs to Drive
    print('\nMoving CSVs to Drive...')
    for f in ['Data_Entry_2017.csv', 'BBox_List_2017.csv',
              'train_val_list.txt', 'test_list.txt']:
        src = os.path.join(extract_dir, f)
        dst = os.path.join(drive_root, f)
        if os.path.exists(dst):
            print(f'  {f} already in Drive, skipping.')
            continue
        if os.path.exists(src):
            subprocess.run(['cp', src, dst])
            print(f'  {f} copied.')

    # Step 4: Move images to Drive
    existing = len(os.listdir(img_dir))
    if existing > 100000:
        print(f'\nImages already in Drive ({existing:,}), skipping.')
    else:
        print('\nMoving images to Drive (15-20 mins)...')
        for i in range(1, 13):
            folder = f'images_{i:03d}'
            src    = os.path.join(extract_dir, folder, 'images')
            if not os.path.exists(src):
                print(f'  {folder} not found, skipping.')
                continue
            print(f'  Moving {folder}...')
            subprocess.run(f'cp -r {src}/* {img_dir}/', shell=True)
            n = len(os.listdir(img_dir))
            print(f'  Images so far: {n:,}')

    n_images = len(os.listdir(img_dir))
    print(f'\nDone! {n_images:,} images in {img_dir}')


def verify_data(drive_root: str = '/content/drive/MyDrive/data/nih'):
    """Quick check that all expected files are present."""
    img_dir = os.path.join(drive_root, 'images')
    checks = {
        'Data_Entry_2017.csv': os.path.join(drive_root, 'Data_Entry_2017.csv'),
        'BBox_List_2017.csv' : os.path.join(drive_root, 'BBox_List_2017.csv'),
        'train_val_list.txt' : os.path.join(drive_root, 'train_val_list.txt'),
        'images/ folder'     : img_dir,
    }
    all_good = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        print(f'  {"checkmark" if exists else "x"} {name}')
        if not exists:
            all_good = False
    n = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
    print(f'  {"ok" if n > 0 else "missing"} {n:,} images found')
    if n == 0:
        all_good = False
    print()
    if all_good:
        print('All good - ready to train!')
    else:
        print('Something missing - run download_nih() again.')
