import os
import zipfile
import urllib.request

MODEL_URL = "https://github.com/francesco481/Solaris/releases/download/v2/best.zip"
CHECKPOINT_DIR = "checkpoints"
ZIP_PATH = os.path.join(CHECKPOINT_DIR, "best.zip")
PTH_PATH = os.path.join(CHECKPOINT_DIR, "best.pth")

def download_model():
    """Descarcă modelul din GitHub Releases dacă nu există local."""
    if os.path.exists(ZIP_PATH) or os.path.exists(PTH_PATH):
        print("[INFO] Modelul există deja")
        return

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"[INFO] Se descarca modelul de la {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, ZIP_PATH)
    print(f"[INFO] Model salvat în {ZIP_PATH}")

def extract_model():
    """Dezarhivează modelul dacă nu este deja extras."""
    if os.path.exists(PTH_PATH):
        print("[INFO] best.pth există deja")
        return

    print(f"[INFO] Se dezarhiveaza {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(CHECKPOINT_DIR)
    print(f"[INFO] Model dezarhivat în {CHECKPOINT_DIR}")

def main():
    download_model()
    extract_model()

if __name__ == "__main__":
    main()
