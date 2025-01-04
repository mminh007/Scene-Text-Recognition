import xml.etree.ElementTree as ET
import os
import shutil
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import mlflow



def extract_data_from_xml(src_dir):
    word_tree = ET.parse(src_dir)
    word_root = word_tree.getroot()

    img_paths = []
    img_sizes = []
    img_labels = []
    bboxes = []

    for img in word_root:
        bbs_of_img = []
        labels_of_img = []
        for bbs in img.findall("taggedRectangles"):
            for bb in bbs:
                if not bb[0].text.isalnum():
                    continue

                if ' ' in bb[0].text.lower() or " " in bb[0].text.lower():
                    continue

                bbs_of_img.append([
                    float(bb.attrib["x"]),
                    float(bb.attrib["y"]),
                    float(bb.attrib["width"]),
                    float(bb.attrib["height"])
                ])
                labels_of_img.append(
                    bb[0].text.lower()
                )
        img_path = os.path.join(src_dir, img[0].text)
        img_paths.append(img_path)
        img_sizes.append(((int(img[1].attrib["x"])), int(img[1].attrib["y"])))

        img_labels.append(labels_of_img)
        bboxes.append(bbs_of_img)

    return img_paths, img_sizes, img_labels, bboxes


def convert_to_yolo_format(img_paths, img_sizes, bboxes):
    yolo_format = []
    for img_path, img_size, bboxes in zip(img_paths, img_sizes, bboxes):
        width, height = img_size

        yolo_labels = []

        for bbox in bboxes:
            x, y, w, h = bbox

            center_x = (x + w/2) / width
            center_y = (y + h/2) / height
            yolo_w = w / width
            yolo_h = h / height

            class_id = 0

            yolo_label = f"{class_id} {center_x} {center_y} {yolo_w} {yolo_h}"
            yolo_labels.append(yolo_label)

        yolo_format.append((img_path, "\n".join(yolo_labels)))

    return yolo_format


def save_data(data, src_img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for img_path, labels in data:
        shutil.copy(os.path.join(src_img_dir, img_path), os.path.join(save_dir, "images"))

        im_name = os.path.basename(img_path)
        img_name = os.path.splitext(im_name)[0]

        with open(os.path.join(save_dir, "labels", f"{img_name}.txt"), "w") as f:
            for label in labels:
                f.write(label + "\n")


def split_bboxes(img_paths, img_labels, bboxes, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    labels = []

    for img_path, img_label, bbs in zip(img_paths, img_labels, bboxes):
        img = Image.open(img_path)

        for label, bb in zip(img_label, bbs):
            cropped_img = img.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))

            if np.mean(cropped_img) < 35 or np.mean(cropped_img) > 220:
                continue

            if cropped_img.size[0] < 10 or cropped_img.size[1] < 10:
                continue

            cropped_img.save(os.path.join(save_dir, f"{count}.jpg"))
            new_img_path = os.path.join(save_dir, f"{count}.jpg")

            label = new_img_path + "\t" + label
            labels.append(label)
            count += 1

    with open(os.path.join(save_dir, "labels.txt"), "w") as f:
        for label in labels:
            f.write(label + "\n")


def encode(label, char_to_idx, max_label_len):
    encoded_label = torch.tensor(
        [char_to_idx[char] for char in label],
        dtype = torch.int32
    )
    label_len = len(encoded_label)
    lengths = torch.tensor(label_len, dtype=torch.int32)

    padded_labels = F.pad(
        encoded_label,
        (0, max_label_len - label_len),
        value = 0
    )

    return padded_labels, lengths


def decode(encoded_sequences, idx_to_char, blank_char="-"):
    decoded_sequences = []

    for seq in encoded_sequences:
        decoded_label = []
        prev_char = None

        for token in seq:
            if token != 0:
                char = idx_to_char[token.item()]

            if char != blank_char:
                if char != prev_char or prev_char == blank_char:
                    decoded_label.append(char)

            prev_char = char
        decoded_sequences.append("".join(decoded_label))

    return decoded_sequences


def create_voc(args):

    SRC_DIR = Path(args.src_dir)
    OCR_DIR = SRC_DIR / args.ocr_dir 

    imgs = []
    labels = []

    with open( OCR_DIR / "labels.txt", "r") as f:
        for line in f:
            img_path, label = line.strip().split("\t")
            imgs.append(img_path)
            labels.append(label)

    letters = [char.split(".")[0].lower() for char in labels]
    letters = "".join(letters)
    letters = sorted(list(set(list(letters))))

    chars = "".join(letters)
    chars += "-"
    vocab_size = len(chars)

    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    max_label_len = max([len(label) for label in labels])

    return imgs, labels, char_to_idx, idx_to_char, max_label_len, vocab_size


def connect_mflow(args, model_name=None):
    
    MLFLOW_TRACKING_URI = args['tracking_uri']

    if model_name == "yolo":
        MLFLOW_EXPERIMENT_NAME = args['yolo_experiment_name']
    else:
        MLFLOW_EXPERIMENT_NAME = args['crnn_experiment_name']

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT_NAME)
        print(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")
        print(f"MLFLOW_EXPERIMENT_NAME: {MLFLOW_EXPERIMENT_NAME}")
    except Exception as e:
        print(f"Error: {e}")
        raise e