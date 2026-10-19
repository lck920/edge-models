"""Run parking detection + plate recognition on the Coral Dev Board Mini.

Usage:   python3 edge_tpu_detection.py photo.jpg [more photos ...]
Needs:   the two *_edgetpu.tflite files from Step 21 of parking_license_merged.ipynb, next to this script.
Install: see "Running on the Coral Dev Board Mini" in README.md.

No training happens here: the models are trained and converted once in the notebook.
Prints one JSON per photo: the owner notifications (Step 19) and the dashboard feed (Step 20).

Written for the board's Python 3.7 with only tflite_runtime, NumPy, Pillow and Tesseract
(Ultralytics and EasyOCR do not run on the board), so the YOLO output is decoded here with NumPy.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime

import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate

PARKING_MODEL_FILE = "parking_detection_yolov8n_edgetpu.tflite"
PLATE_MODEL_FILE = "license_plate_yolov8n_edgetpu.tflite"
PARKING_CONF = 0.35
PARKING_NMS_IOU = 0.7   # YOLO's default
PLATE_CONF = 0.25
PLATE_NMS_IOU = 0.3
CARPARK_LOCATION = "Level 1, Car Park A"  # describes this camera/zone for the dashboard


# --- Models ------------------------------------------------------------------
def load_model(model_file):
    """Load a YOLO .tflite model. *_edgetpu.tflite files run on the Edge TPU, other .tflite files on the CPU."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_file)
    delegates = [load_delegate("libedgetpu.so.1")] if model_file.endswith("_edgetpu.tflite") else []
    interpreter = Interpreter(model_path=path, experimental_delegates=delegates)
    interpreter.allocate_tensors()
    # YOLO stores the class names inside the .tflite file, as {"0": "Empty", "1": "Occupied", ...}
    with zipfile.ZipFile(path) as z:
        names = json.loads(z.read(z.namelist()[0]))["names"]
    interpreter.class_names = [names[str(i)] for i in range(len(names))]
    return interpreter


def letterbox(image, size):
    """Resize the image to fit size x size without stretching, and fill the rest with grey (as YOLO does)."""
    ratio = min(size / image.width, size / image.height)
    new_w, new_h = round(image.width * ratio), round(image.height * ratio)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    return np.asarray(canvas), ratio, pad_x, pad_y


def nms(boxes, scores, iou_threshold):
    """Keep the most confident box and drop the boxes that overlap it too much, then repeat."""
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        w = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
        h = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
        overlap = w * h / (areas[i] + areas[rest] - w * h + 1e-9)
        order = rest[overlap <= iou_threshold]
    return keep


def detect(interpreter, image, conf, iou):
    """image: PIL RGB image. Returns one dict per object: class, conf (0-1) and box (x1, y1, x2, y2) in pixels."""
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    size = inp["shape"][1]

    # The model takes an 8-bit image: pixel / 255, then quantized with the input scale and zero point
    pixels, ratio, pad_x, pad_y = letterbox(image, size)
    scale, zero_point = inp["quantization"]
    x = np.round(pixels / 255.0 / scale + zero_point)
    info = np.iinfo(inp["dtype"])
    interpreter.set_tensor(inp["index"], np.clip(x, info.min, info.max).astype(inp["dtype"])[None])
    interpreter.invoke()

    # Output: (4 + classes) rows x 5376 candidate boxes. Rows 0-3 are the box centre x, centre y, width, height
    # (0-1 of the input size), the other rows are the score of every class.
    scale, zero_point = out["quantization"]
    y = (interpreter.get_tensor(out["index"])[0].astype(np.float32) - zero_point) * scale
    class_scores = y[4:].T
    class_ids = class_scores.argmax(1)
    confs = class_scores.max(1)
    keep = confs >= conf
    cx, cy, w, h = (y[:4, keep] * size)
    class_ids, confs = class_ids[keep], confs[keep]

    # Box corners, moved back from the letterboxed 512 x 512 input to the original photo
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) / ratio, 0, image.width)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) / ratio, 0, image.height)

    # NMS per class: shifting every class to its own area keeps boxes of different classes apart
    kept = nms(boxes + class_ids[:, None] * 10000.0, confs, iou)
    return [{"class": interpreter.class_names[int(class_ids[i])], "conf": float(confs[i]),
             "box": tuple(float(v) for v in boxes[i])} for i in kept]


parking_model = load_model(PARKING_MODEL_FILE)
plate_model = load_model(PLATE_MODEL_FILE)


# --- Plate OCR (Tesseract) -----------------------------------------------------
PLATE_FORMAT = re.compile(r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]?$")
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8", "T": "7"}


def read_plate(crop):
    """crop: PIL image of the plate. Returns the raw text Tesseract reads (top line first)."""
    gray = crop.convert("L")
    if gray.height < 120:  # Tesseract reads small text badly, so enlarge the plate first
        factor = 120 / gray.height
        gray = gray.resize((round(gray.width * factor), 120), Image.BICUBIC)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        gray.save(f.name)
        result = subprocess.run(["tesseract", f.name, "stdout", "--psm", "6",
                                 "-c", "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
    return "".join(result.stdout.split())


def clean_plate(text):
    text = re.sub(r"[^A-Z0-9]", "", text.upper())

    # Try every way to split the text into letters + numbers + optional letter,
    # fix look-alike characters, and keep the split that needs the fewest changes
    best, best_changes = text, None
    for n_prefix in (3, 2, 1):
        for n_suffix in (0, 1):
            prefix = text[:n_prefix]
            digits = text[n_prefix:len(text) - n_suffix]
            suffix = text[len(text) - n_suffix:]

            new_prefix = "".join(TO_LETTER.get(c, c) for c in prefix)
            new_digits = "".join(TO_DIGIT.get(c, c) for c in digits)
            new_suffix = "".join(TO_LETTER.get(c, c) for c in suffix)
            changes = sum(a != b for a, b in zip(prefix + digits + suffix, new_prefix + new_digits + new_suffix))

            # Malaysian plate numbers never start with 0 (an extra 0 is usually the edge of the plate)
            changes += len(new_digits) - len(new_digits.lstrip("0"))
            new_digits = new_digits.lstrip("0")

            candidate = new_prefix + new_digits + new_suffix
            if PLATE_FORMAT.match(candidate) and (best_changes is None or changes < best_changes):
                best, best_changes = candidate, changes
    return best


def find_plate(text):
    """Tesseract often reads the plate border too ("1VBP3477", "VBP34777OS"), so look for the plate inside the text.
    Each part of the text is cleaned with clean_plate(); the part that needs the fewest dropped and fixed characters
    wins (ties: the fewest fixed characters). Returns "" if no part looks like a plate."""
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    best, best_cost = "", None
    for start in range(len(text)):
        for end in range(start + 1, len(text) + 1):
            part = text[start:end]
            plate = clean_plate(part)
            if not PLATE_FORMAT.match(plate):
                continue
            fixed = sum(a != b for a, b in zip(part, plate)) + abs(len(part) - len(plate))
            cost = (fixed + len(text) - len(part), fixed)
            if best_cost is None or cost < best_cost:
                best, best_cost = plate, cost
    return best


def format_plate(plate):
    # VBM6529 -> VBM 6529, WC679T -> WC 679 T
    match = re.match(r"^([A-Z]+)([0-9]+)([A-Z]?)$", plate)
    return " ".join(part for part in match.groups() if part) if match else plate


def read_plate_number(car):
    """car: PIL image of one car. Returns the plate ("VBM 6529"), or "" if not found or not readable."""
    plates = detect(plate_model, car, PLATE_CONF, PLATE_NMS_IOU)
    if not plates:
        return ""
    x1, y1, x2, y2 = (int(v) for v in max(plates, key=lambda p: p["conf"])["box"])  # the most confident plate
    plate = find_plate(read_plate(car.crop((max(0, x1 - 3), max(0, y1 - 3), x2 + 3, y2 + 3))))
    return format_plate(plate) if plate else ""


# --- Outputs (same format as the notebook, Steps 19 and 20) -------------------
STATUS_COLOR = {"Empty": "#21c25e", "Occupied": "#e03131", "Illegal": "#f08c00"}
STATUS_LABEL = {"Empty": "Free", "Occupied": "Occupied", "Illegal": "Double-parked"}


def process(image_path):
    image = Image.open(image_path).convert("RGB")
    detections = detect(parking_model, image, PARKING_CONF, PARKING_NMS_IOU)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Parking first, then the plate of every double-parked car
    events = []
    for obj in (d for d in detections if d["class"] == "Illegal"):
        x1, y1, x2, y2 = (int(v) for v in obj["box"])
        plate = read_plate_number(image.crop((x1, y1, x2, y2))) or None
        events.append({
            "event": "double_parking_detected",
            "vehicle_id": "V{:03d}".format(len(events) + 1),
            "plate": plate,                 # None if the plate could not be read
            "owner_lookup": "pending" if plate is None else "resolve_in_backend",
            "location": CARPARK_LOCATION,
            "box": [x1, y1, x2, y2],
            "confidence": round(obj["conf"], 3),
            "detected_at": now,
            "message": "Your vehicle is double-parked and blocking the aisle. Please move it.",
            "penalty": "RM 50 (simulated summons)",
            "notify_channel": ["sms", "email", "push"],
        })

    boxes = [{
        "id": "B{:03d}".format(i + 1),
        "status": obj["class"],
        "label": STATUS_LABEL[obj["class"]],
        "color": STATUS_COLOR[obj["class"]],
        "parkable": obj["class"] == "Empty",
        "confidence": round(obj["conf"], 3),
        "box": [round(v) for v in obj["box"]],
    } for i, obj in enumerate(detections)]
    dashboard = {
        "location": CARPARK_LOCATION,
        "generated_at": now,
        "summary": {name.lower(): sum(b["status"] == name for b in boxes) for name in ("Empty", "Occupied", "Illegal")},
        "boxes": boxes,
    }
    return {"image": image_path, "notifications": events, "dashboard": dashboard}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 edge_tpu_detection.py photo.jpg [more photos ...]")
    for path in sys.argv[1:]:
        print(json.dumps(process(path), indent=2))
