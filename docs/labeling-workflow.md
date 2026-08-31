# Mouse and bottle labeling workflow

This project uses your own photos. The public tool below only creates initial boxes; every image needs your review before training.

## 1. Put your photos in the local raw folder

Copy all your mouse, bottle, together and occluded images into `raw/`. They are intentionally ignored by Git so personal photos are not published.

## 2. Create a Windows Python 3 environment

Use Python 3.10–3.12, not the computer's old Python 2.7 installation.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 3. Generate initial labels

The following MIT-licensed project is used solely for automatic pre-labeling. It does not supply this project's data or trained model.

```powershell
git clone https://github.com/hanlinji4002/desk-object-labeling.git work/desk-object-labeling
pip install -r work/desk-object-labeling/requirements.txt
python work/desk-object-labeling/tools/auto_label.py --images raw --out labeled --classes mouse bottle --model yolov8x.pt --conf 0.25
```

The first run downloads `yolov8x.pt`. Keep the class order exactly `mouse`, then `bottle`.

## 4. Review every label in LabelImg

```powershell
pip install labelImg
labelImg labeled/images
```

Select YOLO format. For every image, check all of the following:

- `mouse` is class `0`; `bottle` is class `1`.
- Each visible object has its own tight rectangular box.
- Remove false boxes, repair boxes that are shifted or too large, and add missed objects.
- A photo containing both objects needs two boxes.
- A partly hidden object is labeled around its visible extent; do not invent an invisible full shape.
- A photo with neither target must have an intentionally empty `.txt` label file.

## 5. Audit and split after review

Run this command only after every image is reviewed:

```powershell
$env:PYTHONPATH = "src"
python -m nb_vision.dataset_tools audit --images labeled/images --labels labeled/labels
```

It must print `Label audit passed`. Then place reviewed image/label pairs together in `labeled/pairs/` and run:

```powershell
python -m nb_vision.dataset_tools split --input labeled/pairs --output dataset --seed 49
```

Keep near-duplicate burst images in a single split. Record actual train/validation/test counts in the project README after the split finishes.
