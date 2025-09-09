# Solaris

This repository contains a U-Net implementation for roof segmentation. The trained model weights are hosted in GitHub Releases.  

## First time run tutorial for unet

After cloning, first run:
```
cd project/trainer
python3 prepare_model.py
python3 unet.py --predict /path/to/image
```
You only have to run `prepare_model.py` once.

## Pipeline run

Assuming `prepare_model.py` was executed.
```
python3 detect_roofs.py --weights checkpoints/best.pth --image /path/to/img --outdir /path/to/output/dir [--verbose]
```

Example
```
python3 detect_roofs.py --weights checkpoints/best.pth --image ../../dataset/test/tile_lat_-33-855365_lng_151-097497_png.rf.e935e42698bb8799d37cd5df04fd21ff.jpg --outdir res
```

Note: The `--verbose` argument plots the bitmask after the CCA and the roofs overlaid with their scores.

## Scores

The score \( s \) for a roof is calculated as:

$$
s = 0.4 \cdot \frac{A}{A_{\text{max}}} + 0.4 \cdot L + 0.2 \cdot C
$$

where:

- \( A \) = area of the roof (in pixels)  
- \( A_{\text{max}} \) = maximum area among all detected roofs (used for normalization)  
- \( L \) = normalized average brightness inside the roof (V channel in HSV, so in the range [0,1])  
- \( C \) = compactness/circularity of the roof (calculated as $\( C = \frac{4 \pi A}{P^2} \)$, theoretical values approximately in [0,1])  

The coefficients (0.4, 0.4, 0.2) are weights that sum to 1 and define the relative importance of each component.

## Output

The COCO JSON format uses the following top-level keys:

* `info`: Dataset-level metadata (free text; optional).

* `licenses`: Optional list of licenses.

* `images`: A list of image entries. Each image entry includes:
    * `id` (`int`): The image identifier.
    * `width` (`int`): The image width.
    * `height` (`int`): The image height.
    * `file_name` (`string`): The image file name.

* `annotations`: A list of annotation entries. Each annotation contains:
    * `id` (`int`): A unique annotation ID.
    * `image_id` (`int`): The ID of the image this annotation belongs to.
    * `category_id` (`int`): The category class (e.g., `1` = roof).
    * `segmentation`: Polygon(s) as a list of lists of `[x1, y1, x2, y2, ...]` floats (coords of pixels along the border).
    * `area` (`float`): The mask area in pixels.
    * `bbox`: A bounding box in the format `[x_min, y_min, width, height]` (integers).
    * `iscrowd`: `0` for a single instance or `1` for a crowd of instances.

* `categories`: A list of categories.
    * Example: `{"id": 1, "name": "roof", "supercategory": "structure"}`

### Example

```json
{
  "info": {
    "description": "Roof instance dataset"
  },
  "licenses": [],
  "images": [
    {
      "id": 1,
      "width": W,
      "height": H,
      "file_name": "input.jpg"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [
        [x1, y1, x2, y2, ...]
      ],
      "area": 1234,
      "bbox": [x, y, w, h],
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "roof",
      "supercategory": "structure"
    }
  ]
}
```

The pipeline also outputs the bitmask.


# Requirements
In order to run the unet and pipeline, there are several libraries that have to be installed.
Run `setup.py` first before anything else using:
```
pip install -e .
```
Note: `setup.py` will downgrade your Numpy version (if already installed) to 1.26. 
More recent versions are not compatible with `detect_roofs.py`