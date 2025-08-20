import os
import zipfile
import argparse

def extract_model(archive_path="checkpoints/best.pth.zip", output_dir="checkpoints"):
    pth_path = os.path.join(output_dir, "best.pth")

    if not os.path.exists(archive_path):
        print(f"[EROARE] Arhiva nu a fost gasita: {archive_path}")
        return
    
    if os.path.exists(pth_path):
        print(f"[INFO] {pth_path} exista deja")
        return
    
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"[INFO] Arhiva {archive_path} a fost dezarhivata in {output_dir}")
    print(f"[INFO] Modelul best.pth este pregatit pentru a fi folosit de U-Net.py")


def prepare_for_push(output_dir="checkpoints"):
    pth_path = os.path.join(output_dir, "best.pth")
    if os.path.exists(pth_path):
        os.remove(pth_path)
        print(f"[INFO] Fisierul {pth_path} a fost sters")
    else:
        print(f"[INFO] {pth_path} nu exista, nimic de sters.")
    print(f"[INFO] Modelul este pregatit pentru push")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pregateste modelul pentru rulare sau push")
    parser.add_argument("--push", action="store_true", help="Sterge best.pth dar pastreaza arhiva pentru push in repo")
    args = parser.parse_args()

    if args.push:
        prepare_for_push()
    else:
        extract_model()
