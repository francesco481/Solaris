# Solaris

This repository contains a U-Net implementation for roof segmentation. The trained model weights are hosted in GitHub Releases.  

## First time run tutorial

After cloning, first run:
```
cd project/trainer
python3 prepare_model.py
python3 U-Net.py
python3 U-Net.py --predict /path/to/image
```
Example
```
 python3 U-Net.py --predict /mnt/d/Rospin/Roofs.v2i.coco/Solaris/dataset/test/tile_lat_-33-865006_lng_151-115402_png.rf.3a2ea468ad9e2207833eac9a15f8a723.jpg
```

You only have to run `prepare_model.py` once.