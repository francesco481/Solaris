import argparse
import os
import random
from typing import List
import io
from urllib.parse import urlparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

from tqdm import tqdm

class SmartRoofMaskDataset(Dataset):
    """
    Dataset care incarca date fie de pe local fie din GCS.
    """
    def __init__(self, img_dir: str, mask_dir: str, img_size: int, exts: List[str] = [".jpg", ".jpeg", ".png"]):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = img_size
        self.is_gcs = img_dir.startswith("gs://")

        if self.is_gcs:
            self.storage_client = storage.Client()
            self.img_bucket_name, self.img_prefix = self._parse_gcs_path(img_dir)
            self.mask_bucket_name, self.mask_prefix = self._parse_gcs_path(mask_dir)
            self.images = self._list_gcs_files(self.img_bucket_name, self.img_prefix, exts)
        else:
            self.images = [f for f in sorted(os.listdir(img_dir)) if os.path.splitext(f)[1].lower() in exts]

        if not self.images:
            raise RuntimeError(f"Nicio imagine gasita in {img_dir}")

        self.tx_img = T.Compose([T.ToTensor(), T.Resize((img_size, img_size), antialias=True)])
        self.tx_mask = lambda m: torch.from_numpy(
            np.array(m.resize((img_size, img_size), resample=Image.NEAREST))
        ).long()

    def _parse_gcs_path(self, gcs_path: str):
        parsed = urlparse(gcs_path)
        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip('/')
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        return bucket_name, prefix

    def _list_gcs_files(self, bucket_name, prefix, exts):
        blobs = self.storage_client.list_blobs(bucket_name, prefix=prefix)
        return [os.path.basename(blob.name) for blob in blobs if os.path.splitext(blob.name)[1].lower() in exts and blob.name != prefix]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        fname = self.images[idx]
        base = os.path.splitext(fname)[0]

        if self.is_gcs:
            # Incarca din GCS
            img_blob = self.storage_client.bucket(self.img_bucket_name).blob(f"{self.img_prefix}{fname}")
            img_bytes = img_blob.download_as_bytes()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            mask_fname = f"{base}_sem.png"
            mask_blob = self.storage_client.bucket(self.mask_bucket_name).blob(f"{self.mask_prefix}{mask_fname}")
            try:
                mask_bytes = mask_blob.download_as_bytes()
                mask = Image.open(io.BytesIO(mask_bytes))
            except Exception as e:
                raise FileNotFoundError(f"Lipsa masca gs://{self.mask_bucket_name}/{self.mask_prefix}{mask_fname}") from e
        else:
            # Incarca Local
            img_path = os.path.join(self.img_dir, fname)
            img = Image.open(img_path).convert("RGB")
            
            mask_path = os.path.join(self.mask_dir, f"{base}_sem.png")
            if not os.path.isfile(mask_path):
                raise FileNotFoundError(f"Lipsă mască pentru {fname}: {mask_path}")
            mask = Image.open(mask_path)

        img_t = self.tx_img(img)
        mask_t = self.tx_mask(mask)
        return img_t, mask_t


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, skip_ch, 2, stride=2)
        self.conv = DoubleConv(skip_ch + skip_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet(nn.Module):
    """
    U-Net (3 clase: bg/roof/boundary)
    """
    def __init__(self, n_classes=3, base=64):
        super().__init__()
        self.inc = DoubleConv(3, base)
        self.down1, self.down2 = Down(base, base*2), Down(base*2, base*4)
        self.down3, self.down4 = Down(base*4, base*8), Down(base*8, base*8)
        self.up1 = Up(base*8, base*8, base*4)
        self.up2 = Up(base*4, base*4, base*2)
        self.up3 = Up(base*2, base*2, base)
        self.up4 = Up(base, base, base)
        self.outc = nn.Conv2d(base, n_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# Utils
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# Train loop
def main():
    ap = argparse.ArgumentParser("Train U-Net on bitmasks (Local/GCS)")
    ap.add_argument("--img_dir", required=True, help="Calea catre folderul cu imagini (locala sau gs://).")
    ap.add_argument("--mask_dir", required=True, help="Calea catre folderul cu masti (locala sau gs://).")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--weights", default="checkpoints/best.pth", help="Calea pentru a salva/incarca checkpoint.")
    ap.add_argument("--resume", action="store_true", help="Daca exista checkpoint, reia antrenarea.")
    ap.add_argument("--gcs_bucket", default=None, help="Bucket-ul GCS unde se salvează modelul.")
    ap.add_argument("--class_weights", default=None, help="Ex: '1.0,1.0,2.0'.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Dispozitivul de utilizat (default: auto).")
    args = ap.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    set_seed(42)

    #Selectare Dispozitiv
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[AVERTISMENT] CUDA nu este disponibil, se va folosi CPU.")
        device = "cpu"
    print(f"[INFO] Se utilizeaza dispozitivul: {device}")
    
    #Incarcare Date
    print("[INFO] Se incarca setul de date...")
    ds = SmartRoofMaskDataset(args.img_dir, args.mask_dir, args.size)
    print(f"[INFO] Au fost gasite {len(ds)} imagini.")
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    pin_memory = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    model = UNet(n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.class_weights:
        w = torch.tensor([float(x) for x in args.class_weights.split(",")], dtype=torch.float32, device=device)
        assert len(w) == 3, "Trebuie trei greutati: bg, roof, boundary"
        ce = nn.CrossEntropyLoss(weight=w)
    else:
        ce = nn.CrossEntropyLoss()

    best_val = float("inf")
    start_epoch = 1

    if args.resume and os.path.isfile(args.weights):
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        best_val = ckpt.get("best_val", best_val)
        start_epoch = ckpt.get("epoch", 1) + 1
        print(f"[INFO] Reluam antrenarea de la epoca {start_epoch}, best_val={best_val:.4f}")

    print("[INFO] Incepe antrenarea...")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs, masks = imgs.to(device, non_blocking=pin_memory), masks.to(device, non_blocking=pin_memory)
            opt.zero_grad()
            logits = model(imgs)
            loss = ce(logits, masks)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device, non_blocking=pin_memory), masks.to(device, non_blocking=pin_memory)
                logits = model(imgs)
                loss = ce(logits, masks)
                v_loss += loss.item()
        v_loss /= len(val_loader)
        print(f"[E{epoch}] Train Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f}")

        # Cloud checkpoint save
        if v_loss < best_val:
            best_val = v_loss
            
            model_path = "checkpoints/best.pth"
            
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "best_val": best_val
            }, model_path)
            print("Model salvat local.")
            
            output_gcs_path = f"gs://unet-training-data/models/best_epoch_{epoch}.pth"
            
            #Verificare salvare in cloud
            try:
                import subprocess
                subprocess.run(["gsutil", "cp", model_path, output_gcs_path], check=True)
                print(f"Modelul a fost copiat in Cloud Storage: {output_gcs_path}")
            except subprocess.CalledProcessError as e:
                print(f"Eroare la copierea modelului in Cloud Storage: {e}")

            # Remove the local file to clean up
            os.remove(model_path)

    print("[DONE] Antrenare incheiata.")

if __name__ == "__main__":
    main()