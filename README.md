# Parking Detection and Plate Recognition

Two YOLOv8n models for car park images.

## ParkingDetection.ipynb

Detects empty parking slots, parked cars and illegally parked cars.

After detection, each car is checked against the parking slots defined for the camera. A car is marked as a parking violation if it overlaps two or more slots, crosses the slot boundary, or is detected as illegally parked by the model.

The parking slots must be defined once in the `SLOTS` setting in Step 14. The values in the notebook only fit the demo image.

## PlateRecognition.ipynb

Detects the licence plate in a photo of a car, then reads the plate number with EasyOCR. The result is corrected using the Malaysian plate format (letters, then numbers, then an optional letter).

## TrashClassification.ipynb

Not part of the parking models. It is kept as a format reference: both notebooks above follow its 17-step structure.

## How to run

1. Compress the `datasets` folder into `datasets.zip`.
2. Open the notebook in Google Colab and set the runtime to T4 GPU.
3. Drag `datasets.zip` into the Files panel on the left.
4. Run all cells.

To run locally instead, keep the notebooks next to the `datasets` folder and install the libraries with `pip install ultralytics easyocr`.

## Datasets

| Folder | Used by | Source | License |
|---|---|---|---|
| `parking-camera-data` | ParkingDetection | [2] | CC BY 4.0 |
| `illegal-parking-data-1` | ParkingDetection | [3] | CC BY-NC-SA 4.0 |
| `illegal-parking-data-2` | ParkingDetection | [4] | CC BY 4.0 |
| `car-plate-data` | PlateRecognition | [1] | CC BY 4.0 |

`illegal-parking-data-1` is licensed for non-commercial use only.

## References

1. malaysia car plate number Computer Vision Model, by gocar. https://universe.roboflow.com/gocar/malaysia-car-plate-number
2. Carpark and License Plate Computer Vision Model, by chung-yi-lai. https://universe.roboflow.com/chung-yi-lai/carpark-and-license-plate
3. Illegal Parking Computer Vision Model, by parking-amu50. https://universe.roboflow.com/parking-amu50/illegal-parking
4. parking slots Computer Vision Dataset, by fatema-yusuf-gkrnh. https://universe.roboflow.com/fatema-yusuf-gkrnh/parking-slots-ebx5s
5. Garbage Classification (used by TrashClassification.ipynb), by mostafaabla, Kaggle. https://www.kaggle.com/datasets/mostafaabla/garbage-classification
6. Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
7. EasyOCR. https://github.com/JaidedAI/EasyOCR
