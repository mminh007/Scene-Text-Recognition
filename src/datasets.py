from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

class STRDataset(Dataset):
  def __init__(self, X, y, char_to_idx, max_label_len, label_encoded = None, transform = None):
    self.img_paths = X
    self.img_labels = y
    self.char_to_idx = char_to_idx
    self.max_label_len = max_label_len
    self.transform = transform
    self.label_encoded = label_encoded

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    img_path = self.img_paths[idx]
    label = self.labels[idx]
    img = Image.open(img_path).convert("RGB")

    if self.transform:
      img = self.transform(img)

    if self.label_encoded:
      label, label_len = self.label_encoded(label, self.char_to_idx, self.max_label_len)

    return img, label, label_len
  

def data_transforms(imgsz: tuple,
                    is_train=None):
  
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(imgsz[0], imgsz[1]),
                transforms.ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                ),
                transforms.Grayscale(
                    num_output_channels=1
                ),
                transforms.GaussianBlur(3),
                transforms.RandomAffine(
                    degrees = 1,
                    shear = 1
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.5,
                    p=0.5,
                    interpolation=3
                ),
                transforms.RandomRotation(degrees=2),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(imgsz[0], imgsz[1]),
                transforms.Grayscale(
                    num_output_channels=1
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ]
        )
    }

    if is_train:
      return data_transforms["train"]

    else:
      return data_transforms["val"]


def build_dataloader(X, y, char_to_idx, max_label_len, labeled_encode, is_train, args):
    
    transforming = transforms(args.imgsz, is_train)

    dataset = STRDataset(X, y, char_to_idx, max_label_len, labeled_encode, transforming)
  
    if is_train:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size * 2, shuffle=True)

    return dataloader