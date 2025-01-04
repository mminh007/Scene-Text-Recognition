from config_args import setup_parse, update_config
from src.datasets import build_dataloader
from src.models import CRNN, evaluate
from src.utils import encode, decode, create_voc, connect_mflow
import numpy as np
import torch
import torch.nn as nn
import mlflow
from mlflow.pytorch import log_model
from pathlib import Path
from sklearn.model_selection import train_test_split
import time


def main(args, **kwargs):
    # Process data for yolo model and CRNN model
    
    OCR_DIR = SRC_DIR / args.ocr_dir

    connect_mflow(args)

    # Trainig CRNN model
    imgs, labels, char_to_idx, idx_to_char, max_label_len, vocab_size = create_voc(args)

    X_train, X_val, y_train, y_val = train_test_split(imgs, labels, test_size=args.test_size, random_state=args.seed, shuffle=True)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=args.val_size, random_state=args.seed, shuffle=True)

    trainloader = build_dataloader(X_train, y_train, char_to_idx, max_label_len, encode, args, is_train=True)
    testloader = build_dataloader(X_test, y_test, char_to_idx, max_label_len, encode, args, is_train=False)
    valloader = build_dataloader(X_val, y_val, char_to_idx, max_label_len, encode, args, is_train=False)

    model = CRNN(vocab_size=vocab_size,
                 hidden_size=args.hidden_state,
                 num_layers=args.num_layers,
                 dropout=args.drop_out,
                 unfreeze_layers=args.unfreeze_layers)
    
    criterion = nn.CTCLoss(blank=char_to_idx["-"],
                           zero_infinity=True,
                           reduction="mean")
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=args.epoch * 0.5, gamma=0.1)
    
    with mlflow.start_run(run_name=args.run_name) as run:
        print(f"MLFLOW run_id: {run.info.run_id}")
        print(f"MLFLOW experiment_id: {run.info.experiment_id}")
        print(f"MLFLOW run_name: {run.info.run_name}")


        mlflow.set_tag(
            {
                "Model's version": args.ml_version
            }
        )

        mlflow.log_params(
            {
                "input_size": args.imgsz,
                "batch_size": args.batch_size,
                "hidden_state": args.hidden_state,
                "num_layers": args.num_layers,
                "unfreeze_layers": args.unfreeze_layers,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "device": args.device,
            }
        )

        best_model_info = {
        "model_state_dict": None,
        "optimizer_state_dict": None,
        "best_loss": None,
        "epoch": None
        }

        best_loss = np.inf
        for epoch in range(args.epochs):
            start = time.time()
            batch_losses = []


            model.train()
            for images, labels, label_lens in trainloader:
                inputs = images.to(args.device)
                targets = labels.to(args.device)
                target_lens = label_lens.to(args.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                logits_lens = torch.full(
                    size = (outputs.size(1),),
                    fill_value = outputs.size(0),
                    dtype = torch.long
                ).to(args.device)

                loss = criterion(outputs, targets, logits_lens, target_lens)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_loss = sum(batch_losses) / len(trainloader)
            mlflow.log_metric("training_loss", f"{epoch_loss:.6f}", step=epoch)

            val_loss = evaluate(model, valloader, args.device, criterion)
            mlflow.log_metric("val_loss", f"{val_loss:.6f}", step=epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_info.update(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_loss": val_loss,
                        "epoch_num": epoch
                    }
                )
            print(f"EPOCH {epoch + 1}: \tTrain loss: {epoch_loss:.4f} \tVal loss: {val_loss:.4f}")

            end = time.time()
            scheduler.step()
            print(f"Time taken: {end - start}")

        SRC_DIR = Path(args.src_dir)
        path_save = SRC_DIR / args.output / run.info.run_id
        path_save.mkdir(parents = True, exist_ok=True)
        
        torch.save(best_model_info, path_save / "best_model.pth")

        # Test loss
        model.load_state_dict(best_model_info["model_state_dict"])
        test_loss = evaluate(model, testloader, device=args.device, criterion=criterion)
        mlflow.log_metric("test loss", f"{test_loss:.6f}", step=float)

        # log model to mlflow model sever
        log_model(model,
                  artifact_path="CRNN-model",
                  pip_requirements= "./requirements")
        
        mlflow.log_artifact("./configs/base_parameters.yaml", artifact_path="config")
        
        kwargs["ti"].xcom_push(key="run_id", value = run.info.run_id)
        kwargs["ti"].xcom_push(key="val_loss", value=best_model_info["best_loss"])

        print("Training Completed!")

        mlflow.end_run()


if __name__ == "__main__":
    parser = setup_parse()

    args = parser.parse_args()
    args = update_config(args)

    main(args)