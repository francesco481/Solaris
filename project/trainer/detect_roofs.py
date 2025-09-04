import os
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Import UNet
from unet import UNet

def run_unet(image_path, weights, device, out_mask_path):
    """Ruleaza U-Net pentru a produce o masca semantica (0=background,1=acoperis,2=frontiera)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(n_classes=3).to(device)
    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = T.Compose([T.ToTensor(), T.Resize((512, 512), antialias=True)])
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).to(device)

    with torch.no_grad():
        logits = model(img_t.unsqueeze(0))
        pred = torch.argmax(logits, dim=1)[0]

    mask = pred.cpu().numpy().astype(np.uint8)
    cv2.imwrite(str(out_mask_path), mask)
    return out_mask_path

def separate_instances(mask_path, min_size=300):
    """Transforma masca semantica in etichete de instante."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"[AVERTISMENT] Fisierul nu exista: {mask_path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Eliminare pixeli frontiera (clasa=2)
    mask_clean = mask.copy()
    mask_clean[mask_clean == 2] = 0
    bw = (mask_clean == 1).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labeled = cv2.connectedComponents(bw)

    final = np.zeros_like(labeled, dtype=np.int32)
    cur = 1
    for lab in range(1, num_labels):
        mask_lab = (labeled == lab).astype(np.uint8)
        if mask_lab.sum() < min_size:
            continue
        final[mask_lab == 1] = cur
        cur += 1

    return final

def mask_to_polygons(binary_mask):
    bm = (binary_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        cnt = cnt.squeeze()
        if cnt.ndim != 2:
            continue
        poly = cnt.flatten().tolist()
        if len(poly) >= 6:
            polys.append([float(x) for x in poly])
    return polys

def instances_to_coco(labeled_mask, image_filename, image_id=1, category_id=1):
    """Converteste masca etichetata in dictionar JSON COCO."""
    h, w = labeled_mask.shape
    images = [{
        "id": image_id,
        "width": int(w),
        "height": int(h),
        "file_name": os.path.basename(image_filename)
    }]

    annotations = []
    ann_id = 1
    for lab in range(1, labeled_mask.max()+1):
        inst_mask = (labeled_mask == lab).astype(np.uint8)
        polys = mask_to_polygons(inst_mask)
        if not polys:
            continue
        ys, xs = np.where(inst_mask)
        bbox = [int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        area = int(inst_mask.sum())
        ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": polys,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0
        }
        annotations.append(ann)
        ann_id += 1

    categories = [{"id": category_id, "name": "roof", "supercategory": "structure"}]

    coco = {
        "info": {"description": "Dataset instante acoperis"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    return coco

def run_pipeline(weights, image_path, outdir, device="cpu", verbose=False):
    os.makedirs(outdir, exist_ok=True)
    mask_path = Path(outdir) / "semantic_mask.png"

    print(f"[INFO] Pasul 1: segmentare semantica")
    run_unet(image_path, weights, device, mask_path)

    print(f"[INFO] Pasul 2: separare instante")
    inst_mask = separate_instances(mask_path)

    print(f"[INFO] Pasul 3: export COCO")
    coco = instances_to_coco(inst_mask, image_path)
    out_json = Path(outdir) / "annotations.json"
    with open(out_json, "w") as f:
        json.dump(coco, f, indent=2)

    # Plot (optional)
    if verbose:
        img = np.array(Image.open(image_path).convert("RGB"))
        cmap = np.random.randint(0, 255, size=(inst_mask.max() + 1, 3), dtype=np.uint8)
        colored_mask = cmap[inst_mask]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(img)
        axs[0].set_title("Imagine originala")
        axs[0].axis("off")

        axs[1].imshow(colored_mask)
        axs[1].set_title("Masca instante")
        axs[1].axis("off")
        plt.show()

    print(f"[INFO] Pipeline finalizat. Rezultatele sunt salvate in {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Pipeline Detectie Acoperis")
    ap.add_argument("--weights", required=True, help="Calea catre greutatile UNet antrenate")
    ap.add_argument("--image", required=True, help="Calea imaginii de intrare")
    ap.add_argument("--outdir", required=True, help="Director de iesire")
    ap.add_argument("--device", default="cpu", help="cpu, cuda sau auto")
    ap.add_argument("--verbose", action="store_true", help="Afiseaza plot-uri")
    args = ap.parse_args()

    run_pipeline(args.weights, args.image, args.outdir,
                 device=args.device, verbose=args.verbose)
