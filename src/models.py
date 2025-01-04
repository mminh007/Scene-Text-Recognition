import timm
import torch.nn as nn
import torch
import time
from tqdm import tqdm
import numpy as np
from pathlib import Path


class CRNN(nn.Module):
  def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.2, unfreeze_layers = 3):
    super().__init__()

    backbone = timm.create_model("resnet152", in_chans=1, pretrained=True)
    modules = list(backbone.children())[:-2]
    modules.append(nn.AdaptiveAvgPool2d((1, None)))
    self.backbone = nn.Sequential(*modules)

    for parameter in self.backbone[-unfreeze_layers:].parameters():
      parameter.requires_grad = True

    self.mapSeq = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(dropout),)

    self.gru = nn.GRU(
        input_size = 512,
        hidden_size = hidden_size,
        num_layers = num_layers,
        bidirectional = True,
        batch_first = True,
        dropout = dropout if num_layers > 1 else 0
    )

    self.layer_norm = nn.LayerNorm(hidden_size * 2)

    self.fc = nn.Sequential(
        nn.Linear(hidden_size * 2, vocab_size),
        nn.LogSoftmax(dim=2)
    )

  @torch.autocast(device_type = "cuda")
  def forward(self, x):
    x = self.backbone(x)
    x = x.permute(0, 3, 1, 2)

    x = x.view(x.size(0), x.size(1), -1) #Flatten
    x = self.mapSeq(x)

    x, _ = self.gru(x)
    x = self.layer_norm(x)
    x = self.fc(x)
    x = x.permute(1, 0, 2)

    return x
  

def evaluate(model, data_loader, device, criterion):
  model.eval()
  losses = []

  with torch.np_grad():
    for images, labels, label_lens in data_loader:
      inputs = images.to(device)
      targets = labels.to(device)
      target_lens = label_lens.to(device)

      outputs = model(inputs)
      logits_lens = torch.full(
          size = (outputs.size(1),),
          fill_value = outputs.size(0),
          dtype = torch.long
      ).to(device)

      loss = criterion(outputs, targets, logits_lens, target_lens)
      losses.append(loss.item())

  return sum(losses) / len(data_loader)

def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, args, run_id, **kwargs):
    train_losses = []
    val_losses = []

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
        for images, labels, label_lens in tqdm(train_loader):
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

        train_loss = sum(batch_losses) / len(train_loader)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, args.device, criterion)
        val_losses.append(val_loss)

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
        print(f"EPOCH {epoch + 1}: \tTrain loss: {train_loss:.4f} \tVal loss: {val_loss:.4f}")

        end = time.time()
        scheduler.step()
        print(f"Time taken: {end - start}")

    SRC_DIR = Path(args.src_dir)
    path_save = SRC_DIR / args.output / run_id
    path_save.mkdir(parents = True, exist_ok=True)
    
    torch.save(best_model_info, path_save / "best_model.pth")
    
    kwargs["ti"].xcom_push(key="run_id", value = run_id)
    kwargs["ti"].xcom_push(key="val_loss", value=best_model_info["best_loss"])
    
    return train_losses, val_losses
