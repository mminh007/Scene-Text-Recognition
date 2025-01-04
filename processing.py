import os
from pathlib import Path
import zipfile
import yaml
from src.utils import extract_data_from_xml, convert_to_yolo_format, save_data, split_bboxes
from sklearn.model_selection import train_test_split
from config_args import setup_parse, update_config


def main(args):
    """
    Processing data for yolo model and recognition model
    """

    SRC_DIR = Path(args.src_dir)
    TMP_DIR = Path(args.tmp_dir)

    DATA_DIR = SRC_DIR / args.data
    OCR_DIR = TMP_DIR / args.ocr_dir

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(OCR_DIR):
        raise FileNotFoundError(f"Failed to create directory: {OCR_DIR}")
    
    
    cmd_0 = f"unzip -q {DATA_DIR} -d {TMP_DIR}" # copy data from local to cloud's folder
    os.system(cmd_0)

    if not os.path.exists(f"{TMP_DIR} / {args.data}"):
        raise FileNotFoundError(f"Data pull {args.data} failed")

    words_file = TMP_DIR / args.xml_file

    img_paths, img_sizes, img_labels, bboxes = extract_data_from_xml(words_file)

    yolo_data = convert_to_yolo_format(img_paths, img_sizes, img_paths, bboxes)

    yolo_train, yolo_test= train_test_split(yolo_data, test_size= args.yl_test_size, random_state=args.yl_seed, shuffle=True)
    yolo_test, yolo_val = train_test_split(yolo_test, test_size=args.yl_val_size, random_state=args.yl_seed, shuffle=True)

    # create YOLO DIR
    YOLO_DIR = TMP_DIR / args.yolo_dir
    YOLO_DIR.mkdir(parents=True, exist_ok=True)

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

    # create data for Text Recognition
    split_bboxes(img_paths, img_labels, bboxes, OCR_DIR)


if __name__ == "__main__":
    parser = setup_parse()

    args = parser.parse_args()
    args = update_config(args)

    main(args)




                     
                                                 

