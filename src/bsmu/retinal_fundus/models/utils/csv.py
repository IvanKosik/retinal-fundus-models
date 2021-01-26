import random
from pathlib import Path

import pandas as pd


def generate_train_valid_csv(image_dir: Path, mask_dir: Path,
                             train_csv_path: Path, valid_csv_path: Path,
                             filter_predicate=lambda file_name: True, train_part: float = 0.75):
    data_file_names = []
    for image_path in image_dir.iterdir():
        if not filter_predicate(image_path.name):
            continue

        mask_path = mask_dir / image_path.name
        if mask_path.exists():
            data_file_names.append(image_path.name)
        else:
            print('WARNING: no mask', mask_path)

    train_file_names = random.sample(data_file_names, int(train_part * len(data_file_names)))
    valid_file_names = [file_name for file_name in data_file_names if file_name not in train_file_names]

    COLUMNS = ['file_names']
    train_data_frame = pd.DataFrame(data=train_file_names, columns=COLUMNS)
    train_data_frame.to_csv(str(train_csv_path), index=False)

    valid_data_frame = pd.DataFrame(data=valid_file_names, columns=COLUMNS)
    valid_data_frame.to_csv(str(valid_csv_path), index=False)
