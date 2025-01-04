import mlflow
from mlflow.pytorch import log_model
from pathlib import Path
from ultralytics import YOLO, settings
from src.utils import connect_mflow
from config_args import setup_parse, update_config


def main(args):

    SRC_DIR = Path(args.src_dir)
    YOLO_DIR = SRC_DIR / args.yolo_dir

    connect_mflow(args, model_name="yolo")
    
    settings.update({
        "mlflow": True
    })
    settings.reset()

    yolo = YOLO("yolo8n.pt")

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tag({
            "model's version": "yolo8n",
        })

        mlflow.log_params({
            "epochs": args.yl_epochs,
            "imgsz": args.yl_imgsz,
            "batch": args.yl_batch,
        })

        results = yolo.train(data = YOLO_DIR / "data.yaml",
                epochs = args.yl_epochs,
                imgsz = args.yl_imgsz,
                batch=args.yl_batch,
                cache = True if args.yl_cache == 1 else False,
                plots = True if args.yl_plots == 1 else False)
        
        mlflow.log_metrics(
            {
                "f1_curve": results.box.f1_curve,
                "P_curve": results.box.p_curve,
                "R_curve": results.box.r_curve,      
            }
        )

        log_model(yolo, "yolo_model")
        # checkpoint = "./runs/detect/train/weights/best.pt"
        # model = YOLO(checkpoint)

        # metrics = model.val()
        mlflow.end_run()


if __name__ == "__main__":
    parser = setup_parse()

    args = parser.parse_args()
    args = update_config(args)

    main(args)