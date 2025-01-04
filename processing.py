import os
from pathlib import Path
import zipfile
import yaml
from src.utils import extract_data_from_xml, convert_to_yolo_format, save_data, split_bboxes
from sklearn.model_selection import train_test_split


def data_processing(args):
    SRC_DIR = Path(args.src_dir)
    DATA_DIR = SRC_DIR / args.data
    

    with zipfile.ZipFile(DATA_DIR,"r") as zip_ref:
        zip_ref.extractall(SRC_DIR)
    
    words_file = SRC_DIR / args.xml_file

    img_paths, img_sizes, img_labels, bboxes = extract_data_from_xml(words_file)

    yolo_data = convert_to_yolo_format(img_paths, img_sizes, img_paths, bboxes)

    yolo_train, yolo_test= train_test_split(yolo_data, test_size= args.yl_test_size, random_state=args.yl_seed, shuffle=True)
    yolo_test, yolo_val = train_test_split(yolo_test, test_size=args.yl_val_size, random_state=args.yl_seed, shuffle=True)

    if args.yl_save_data:
        YOLO_DIR = SRC_DIR / args.yolo_dir
        save_data(yolo_train, os.path.join(YOLO_DIR, "train"))
        save_data(yolo_val, os.path.join(YOLO_DIR, "val"))
        save_data(yolo_test, os.path.join(YOLO_DIR, "test"))
    
    data_yaml = {
        "path": YOLO_DIR,
        "train": "train/images",
        "test": "test/images",
        "val": "val/images",
        "nc": 1,
        "names": "text"
    }

    yolo_yaml_path = os.path.join(YOLO_DIR, "data.yaml")
    with open(yolo_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    OCR_DIR = SRC_DIR / args.ocr_dir

    # create data for Text Recognition
    split_bboxes(img_paths, img_labels, bboxes, OCR_DIR)







                     
                                                 

