# Solaris

This repository contains a U-Net implementation for roof segmentation. The trained model weights are hosted in GitHub Releases.  

## First time run tutorial for unet

After cloning, first run:
```
cd project/trainer
python3 prepare_model.py
python3 unet.py --predict /path/to/image
```
Example
```
 python3 unet.py --predict /mnt/d/Rospin/Roofs.v2i.coco/Solaris/dataset/test/tile_lat_-33-865006_lng_151-115402_png.rf.3a2ea468ad9e2207833eac9a15f8a723.jpg
```

You only have to run `prepare_model.py` once.

## Pipeline run

Assuming `prepare_model.py` was executed.
```
python3 detect_roofs.py --weights checkpoints/best.pth --image /path/to/img --outdir /path/to/output/dir [--verbose]
```

Example
```
python3 detect_roofs.py --weights checkpoints/best.pth --image /mnt/d/Rospin/Solaris/dataset/test/tile_lat_-33-855365_lng_151-097497_png.rf.e935e42698bb8799d37cd5df04fd21ff.jpg --outdir res
```

Note: The `--verbose` argument plots the bitmask after the CCA.

# Requirements
In order to run the unet and pipeline, there are several libraries that have to be installed.
Run `setup.py` first before anything else.
Note: `setup.py` will downgrade your Numpy version (if already installed) to 1.26. 
More recent versions are not compatible with `detect_roofs.py`


TODOS: Modify setup.py, change json structure maybe try and overlay plots for --verbose
