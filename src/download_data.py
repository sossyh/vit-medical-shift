"""
download_data.py
----------------
Downloads NIH ChestX-ray14 dataset from Kaggle directly to Google Drive.
Run once from Colab — never needs to run again.

Usage in Colab:
    from src.download_data import download_nih
    download_nih(drive_root='/content/drive/MyDrive/data/nih')
"""

import os
import subprocess


def setup_kaggle(kaggle_json_path: str = 'kaggle.json'):
    """
    Set up Kaggle credentials from an uploaded kaggle.json file.
    Call files.upload() in Colab first, then pass the path here.
    """
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    dest = os.path.expanduser('~/.kaggle/kaggle.json')
    subprocess.run(['cp', kaggle_json_path, dest])
    subprocess.run(['chmod', '600', dest])
    print('Kaggle credentials set up.')


def download_nih(drive_root: str = '/content/drive/MyDrive/data/nih',
                 n_image_zips: int = 2):
    """
    Download NIH ChestX-ray14 CSVs and image zips to Google Drive.

    Args:
        drive_root:   path to your NIH data folder in Google Drive
        n_image_zips: how many image zips to download (1-12)
                      each zip is ~3.5GB with ~9k images
                      2 zips = ~7GB, ~18k images — enough to start
                      4 zips = ~14GB, ~36k images — recommended for thesis
    """
    os.makedirs(drive_root, exist_ok=True)
    img_dir = os.path.join(drive_root, 'images')
    os.makedirs(img_dir, exist_ok=True)

    base_cmd = [
        'kaggle', 'datasets', 'download',
        '-d', 'nih-chest-xrays/data',
        '--path', drive_root
    ]

    # ── Download CSVs ──────────────────────────────────────────────────────
    print('Downloading CSVs...')
    for f in ['Data_Entry_2017.csv', 'BBox_List_2017.csv',
              'train_val_list.txt', 'test_list.txt']:
        csv_path = os.path.join(drive_root, f)
        if os.path.exists(csv_path):
            print(f'  {f} already exists, skipping.')
            continue
        print(f'  Downloading {f}...')
        subprocess.run(base_cmd + ['-f', f])
        # unzip if needed
        zip_path = csv_path + '.zip'
        if os.path.exists(zip_path):
            subprocess.run(['unzip', '-q', zip_path, '-d', drive_root])
            os.remove(zip_path)

    # ── Download image zips ────────────────────────────────────────────────
    for i in range(1, n_image_zips + 1):
        zip_name = f'images_{i:03d}.tar.gz'
        zip_path = os.path.join(drive_root, zip_name)

        if os.path.exists(zip_path):
            print(f'{zip_name} already downloaded, extracting...')
        else:
            print(f'Downloading {zip_name} (~3.5GB)...')
            subprocess.run(base_cmd + ['-f', zip_name])

        # Extract
        print(f'Extracting {zip_name}...')
        subprocess.run([
            'tar', '-xzf', zip_path,
            '-C', img_dir,
            '--strip-components=1'   # removes the images/ subfolder inside the tar
        ])

        # Remove zip to save Drive space
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f'Removed {zip_name} zip to save space.')

    # ── Summary ───────────────────────────────────────────────────────────
    n_images = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
    print(f'\nDone! {n_images:,} images in {img_dir}')
    print('Files in drive_root:')
    for f in sorted(os.listdir(drive_root)):
        print(f'  {f}')


def verify_data(drive_root: str = '/content/drive/MyDrive/data/nih'):
    """Quick check that all expected files are present."""
    checks = {
        'Data_Entry_2017.csv': os.path.join(drive_root, 'Data_Entry_2017.csv'),
        'BBox_List_2017.csv' : os.path.join(drive_root, 'BBox_List_2017.csv'),
        'images/ folder'     : os.path.join(drive_root, 'images'),
    }
    all_good = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = '✓' if exists else '✗'
        print(f'  {status} {name}')
        if not exists:
            all_good = False

    if os.path.exists(os.path.join(drive_root, 'images')):
        n = len(os.listdir(os.path.join(drive_root, 'images')))
        print(f'  ✓ {n:,} images found')

    if all_good:
        print('\nAll good — ready to train!')
    else:
        print('\nSome files missing — run download_nih() first.')
