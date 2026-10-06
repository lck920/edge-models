# Parking Detection and Plate Recognition

Two YOLOv8n models for a Malaysian car park. They are kept in separate notebooks and are built to be merged into one notebook later.

## ParkingDetection.ipynb

Trained on drone photos of a car park. Detects three classes:

- **Empty**: a free parking slot
- **Occupied**: a parked car in a slot
- **Illegal**: a car double-parked in the driving aisle

A car is marked as a parking violation when the model classifies it as Illegal.

For a fixed camera, each car can also be checked against the parking slots: a car is a violation if it overlaps two or more slots or crosses the slot boundary. The slots must be defined once in the `SLOTS` setting in Step 14. `SLOTS` is empty by default, because a drone moves and its slots are in a different place in every photo.

## PlateRecognition.ipynb

Detects the licence plate in a photo of a car, then reads the plate number with EasyOCR. The result is corrected using the Malaysian plate format (letters, then numbers, then an optional letter).

## Merging the two notebooks

Both notebooks end with a **ready-to-merge** step (Step 18) that only needs the saved model file and uses names unique to that notebook:

| Notebook | Model file | Function |
|---|---|---|
| ParkingDetection | `parking_drone_yolov8n.pt` | `detect_parking(image)`: list of objects with `class`, `conf` and `box` |
| PlateRecognition | `plate_yolov8n.pt` | `read_plate_number(image)`: `plate`, `raw_text`, `ocr_conf` and `box` |

Both functions take an RGB image array (`load_rgb(path)`) and give boxes as `(x1, y1, x2, y2)` in pixels, so a car box from `detect_parking` can be cut out of the image and passed to `read_plate_number`.

To build the combined notebook, copy Step 1 and the "OCR functions" cell (Step 14) from PlateRecognition, the Step 18 cell from both notebooks, and both model files. The earlier cells use the same names in both notebooks (`best_model`, `CONF_THRESHOLD`, `DATASET_PATH`, ...) and are only needed for training.

## TrashClassification.ipynb

Not part of the parking models. It is kept as a format reference: the notebooks above follow its step-by-step structure.

## How to run

1. Compress the `datasets` folder into `datasets.zip`.
2. Open the notebook in Google Colab and set the runtime to T4 GPU.
3. Drag `datasets.zip` into the Files panel on the left.
4. Run all cells.

Training is skipped when the saved model file (`parking_drone_yolov8n.pt` or `plate_yolov8n.pt`) is already next to the notebook. On Colab, download the file after training and drag it back in next time. Delete it to train again.

To run locally instead, keep the notebooks next to the `datasets` folder and install the libraries with `pip install ultralytics easyocr`.

## Datasets

| Folder | Used by | Source | License |
|---|---|---|---|
| `carpark-and-lp-with-illegal-parking` | ParkingDetection | [2] | CC BY 4.0 |
| `car-plate-data` | PlateRecognition | [1] | CC BY 4.0 |

`carpark-and-lp-with-illegal-parking` also labels licence plates. ParkingDetection does not use them.

## References

1. malaysia car plate number Computer Vision Model, by gocar. https://universe.roboflow.com/gocar/malaysia-car-plate-number
2. Carpark and License Plate Computer Vision Model, by chung-yi-lai. https://universe.roboflow.com/chung-yi-lai/carpark-and-license-plate
3. Garbage Classification (used by TrashClassification.ipynb), by mostafaabla, Kaggle. https://www.kaggle.com/datasets/mostafaabla/garbage-classification
4. Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
5. EasyOCR. https://github.com/JaidedAI/EasyOCR
