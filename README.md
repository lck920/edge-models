# Parking Detection and Plate Recognition

Two YOLOv8n models for a Malaysian car park, used together in `parking_license_merged.ipynb`:

- **Parking detection** (`parking_detection_yolov8n.pt`): finds empty slots (**Empty**), parked cars (**Occupied**) and cars double-parked in the driving aisle (**Illegal**).
- **Plate recognition** (`license_plate_yolov8n.pt` + EasyOCR): finds the licence plate of a car and reads it. The result is corrected using the Malaysian plate format (letters, then numbers, then an optional letter).

For every photo, parking detection runs first. Each Illegal car is then cut out of the photo, and the plate model reads its plate. The result is an owner notification and a dashboard feed, both as JSON.

## Files

| File | What it is |
|---|---|
| `parking_license_merged.ipynb` | The main notebook: trains (or loads) the parking model, runs both models, and converts them for the Edge TPU |
| `edge_tpu_detection.py` | The script that runs both models on the Coral Edge TPU device |
| `datasets/` | The datasets and the trained models (`datasets/models`) |
| `base_models/` | The original separate notebooks, only needed to retrain a model (see below) |
| `TrashClassification.ipynb` | Not part of the parking models. Kept as a format reference: the notebooks follow its step-by-step structure |

```
datasets/
├── car-plate-data/                        plate dataset (used by base_models/PlateRecognition.ipynb)
├── carpark-and-lp-with-illegal-parking/   car park dataset (used by the merged notebook)
└── models/
    ├── parking_detection_yolov8n.pt               trained models (used by the notebook)
    ├── license_plate_yolov8n.pt
    ├── parking_detection_yolov8n_edgetpu.tflite   converted models (used on the Edge TPU device)
    └── license_plate_yolov8n_edgetpu.tflite
```

## How to run the notebook

1. Compress the `datasets` folder into `datasets.zip`. Zip the folder itself, not its contents.
2. Open `parking_license_merged.ipynb` in Google Colab and set the runtime to T4 GPU.
3. Drag `datasets.zip` into the Files panel on the left.
4. Run all cells.

The zip holds the dataset and both trained models, so nothing else has to be uploaded. Training is skipped when `datasets/models/parking_detection_yolov8n.pt` exists. Delete it to train again. On Colab, download the new model after training and put it in `datasets/models` before zipping the folder next time.

To run locally instead, keep the notebook next to the `datasets` folder and install the libraries with `pip install ultralytics easyocr`.

### Notebook steps

- **Steps 1-17:** load the car park dataset, train and evaluate the parking model, and test it on one image.
- **Steps 17.1-17.6:** send the model's predictions of the Step 17 image to BigQuery (one row per object: `timestamp`, `image_name`, `predicted_class`, `confidence`, `latency_ms`), for a Looker Studio dashboard. Create the BigQuery table first and set `PROJECT_ID` in Step 17.2.
- **Step 18:** load both models: `detect_parking(image)` and `read_plate_number(image)`.
- **Step 19:** the owner notification for every double-parked car, with its plate number.
- **Step 20:** the dashboard feed (a virtual box per parking area: free, occupied or double-parked).
- **Step 21:** convert both models for the Edge TPU.

The notebook only uses the `carpark-and-lp-with-illegal-parking` dataset and maps its classes by name. Do not swap in the Roboflow download `Carpark-and-License-Plate-1`: it numbers its categories differently, and an older version of this notebook learned to detect licence plates as Illegal because of it.

## Running on the Coral Edge TPU

Training happens once, in the notebook. Nothing is trained on the device.

### 1. Convert the models (only after retraining)

The Edge TPU cannot run `.pt` files. The converted files are already in `datasets/models`, ready to use:

- `parking_detection_yolov8n_edgetpu.tflite`
- `license_plate_yolov8n_edgetpu.tflite`

After retraining a model, convert again with Step 21 of the notebook, on Colab (the Edge TPU compiler only works on x86_64 Linux), and put the downloaded files in `datasets/models`. Step 21 converts in a separate Python 3.11 with TensorFlow 2.19: with Colab's own newer TensorFlow, almost the whole model ends up on the board's CPU instead of the Edge TPU. It prints how many operations run on the Edge TPU (about 238 of 256 is right) and stops if most would run on the CPU. Keep the `_edgetpu.tflite` ending in the file names.

### 2. Set up the Dev Board Mini (once)

The Coral Dev Board Mini is a small computer with the Edge TPU built in. It runs Mendel Linux (based on Debian 10) with Python 3.7, and it already comes with the Edge TPU runtime and `tflite_runtime`. Ultralytics and EasyOCR do not run on it, so `edge_tpu_detection.py` only uses `tflite_runtime`, NumPy, Pillow and Tesseract (for reading the plate).

You need:

- a 5 V / 2 A USB-C power supply
- a USB-C data cable (a charge-only cable does not work)
- a Windows PC with Python 3 and Git for Windows (Git Bash)
- a Wi-Fi network for the board

The board is set up from the PC with MDT (Mendel Development Tool) [3]. The Windows steps follow the buyzero guide [4].

1. **Install MDT on the PC.** In Git Bash:

   ```
   python3 -m pip install --user mendel-development-tool
   echo "alias mdt='winpty mdt'" >> ~/.bash_profile
   source ~/.bash_profile
   ```

   If `mdt` is then "command not found", add the folder pip installed it to (pip prints it in a warning) to the Windows Path.

2. **Connect the board.** Plug the power supply into the USB-C port labelled for power, and the data cable from the other USB-C port (OTG) to the PC. Wait about 30 seconds after the green LED turns on.

3. **Open a shell on the board.** In Git Bash:

   ```
   mdt devices
   mdt shell
   ```

   `mdt devices` must show the board's name. Every command from here to step 6 runs in this `mdt shell`. If `mdt shell` says "Key not present", follow the fix in [4].

4. **Connect the board to Wi-Fi.** Use your network's name (in quotes if it has spaces); `--ask` asks for the password:

   ```
   nmcli --ask dev wifi connect 'NETWORK NAME' ifname wlan0
   ping -c 3 google.com
   ```

5. **Check the Edge TPU:**

   ```
   python3 -c "from tflite_runtime.interpreter import load_delegate; load_delegate('libedgetpu.so.1'); print('Edge TPU OK')"
   ```

6. **Install Tesseract** (NumPy and Pillow are usually installed already):

   ```
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr python3-numpy python3-pil
   tesseract --version
   ```

   If `apt-get update` fails with "404 Not Found" errors, the Debian 10 package servers have moved to the Debian archive. Add the archive and try again:

   ```
   echo "deb http://archive.debian.org/debian buster main" | sudo tee /etc/apt/sources.list.d/debian-archive.list
   sudo apt-get -o Acquire::Check-Valid-Until=false update
   sudo apt-get install -y tesseract-ocr python3-numpy python3-pil
   ```

7. **Copy the files to the board.** Only these three files are needed on the board, plus a photo to test with. The `.pt` files, the notebook and the datasets are not needed.

   | File | Where it is in this project |
   |---|---|
   | `edge_tpu_detection.py` | the project folder |
   | `parking_detection_yolov8n_edgetpu.tflite` | `datasets/models/` |
   | `license_plate_yolov8n_edgetpu.tflite` | `datasets/models/` |

   Type `exit` to leave the board's shell, then in Git Bash, in this project folder:

   ```
   mdt push edge_tpu_detection.py
   mdt push datasets/models/parking_detection_yolov8n_edgetpu.tflite
   mdt push datasets/models/license_plate_yolov8n_edgetpu.tflite
   mdt push photo.jpg
   ```

   The files go to the home folder on the board (`/home/mendel`). `photo.jpg` is any car park photo to test with. The script looks for the two `.tflite` files next to itself, so keep all three in the same folder and do not rename them.

### 3. Run

```
mdt shell
python3 edge_tpu_detection.py photo.jpg
```

More photos can be given at once (`python3 edge_tpu_detection.py a.jpg b.jpg`). For every photo the script prints the owner notifications and the dashboard feed as JSON, in the same format as Steps 19 and 20 of the notebook.

To switch the board off: unplug the data cable, press the power button briefly, and unplug the power supply once the green LED is off.

### What to expect

Tested on a PC with the same 8-bit models on the CPU (not yet on the board itself):

- **Parking detection:** test mAP50 0.975 (the `.pt` model: 0.982).
- **Plate reading:** Tesseract read 9 of the 13 test plates correctly; EasyOCR in the notebook reads all 13. Tesseract mostly misses characters on plates with glare or small text. The board has an older Tesseract (version 4) than the PC test (version 5), so its results may differ a little.

## Retraining with the base models

`base_models/ParkingDetection.ipynb` and `base_models/PlateRecognition.ipynb` are the original separate notebooks. The merged notebook trains the parking model itself, but the plate model is only trained in `base_models/PlateRecognition.ipynb`.

The base notebooks save their models as `parking_drone_yolov8n.pt` and `plate_yolov8n.pt`. To use a retrained model, rename it to `parking_detection_yolov8n.pt` or `license_plate_yolov8n.pt` and put it in `datasets/models`.

For a fixed camera, `base_models/ParkingDetection.ipynb` and the merged notebook (Step 14) can also check every car against the parking slots: a car is a violation if it overlaps two or more slots or crosses the slot boundary. The slots must be defined once in the `SLOTS` setting. `SLOTS` is empty by default, because a drone moves and its slots are in a different place in every photo.

## Datasets

| Folder | Used by | Source | License |
|---|---|---|---|
| `carpark-and-lp-with-illegal-parking` | Parking model | [2] | CC BY 4.0 |
| `car-plate-data` | Plate model | [1] | CC BY 4.0 |

`carpark-and-lp-with-illegal-parking` also labels licence plates. The parking model does not use them.

## References

1. malaysia car plate number Computer Vision Model
https://universe.roboflow.com/gocar/malaysia-car-plate-number

2. Carpark and License Plate with Illegal parking Computer Vision Model
https://universe.roboflow.com/chung-yi-lai/carpark-and-license-plate-with-illegal-parking

3. Get started with the Dev Board Mini, Coral (Google)
https://coral.ai/docs/dev-board-mini/get-started/

4. Coral Dev Board Mini: connect and get started (Windows setup), buyzero
https://buyzero.de/en/blogs/news/coral-dev-board-mini-anschliessen-und-loslegen
