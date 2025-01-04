import yaml
import argparse

def setup_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-file", type=str, help="path to config file")
    parser.add_argument("--data", type=str)
    parser.add_argument("--xml-file", type=str)

    parser.add_argument("--src-dir", type=str, help="using save model")
    parser.add_argument("--yolo-dir", type=str, help="data using yolo model")
    parser.add_argument("--ocr-dir", type=str)

    parser.add_argument("--imgsz", type=tuple, help="size of input")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden-state", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--drop-out", type=float)
    parser.add_argument("--unfreeze-layers", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--val-size", type=float)
    parser.add_argument("--test-size", type=float)
    parser.add_argument("--output", type=str)
    
    parser.add_argument("--yl-imgsz", type=int)
    parser.add_argument("--yl-epochs", type=int)
    parser.add_argument("--yl-cache", type=int)
    parser.add_argument("--yl-patience", type=int)
    parser.add_argument("--yl-plots", type=int)
    parser.add_argument("--yl-save-data", type=int)
    parser.add_argument("--yl-test-size", type=float)
    parser.add_argument("--yl-val-size", type=float)
    parser.add_argument("--yl-seed", type=int)
    parser.add_argument("--yl-batch", type=int)
    
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--ml-version", type=str)


    return parser


def update_config(args: argparse.Namespace):
    if not args.config_file:
        return args
    
    cfg_path = args.config_file + ".yaml" if not args.config_file.endswith(".yaml") else args.config_file

    with open(cfg_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, value in data.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # config_args = argparse.Namespace(**data)
    # args = parser.parse_args(namespace=config_args)
    

    return args

