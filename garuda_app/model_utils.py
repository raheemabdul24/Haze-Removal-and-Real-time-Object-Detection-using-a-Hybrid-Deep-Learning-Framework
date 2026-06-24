import torch
import torch.nn as nn
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from sort import Sort

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

vehicle_classes = ["car","truck","bus","motorcycle","bicycle"]
person_class = "person"


class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class SimpleDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class SWDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimpleEncoder()
        self.decoder = SimpleDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


def load_model():

    model = SWDModel().to(device)

    model_path = os.path.join(SCRIPT_DIR, "models", "dehazing_best_model.pth")
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    return model


def load_yolo():
    yolo_path = os.path.join(SCRIPT_DIR, "yolov8n.pt")
    yolo_model = YOLO(yolo_path)
    return yolo_model


def load_tracker():
    # tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    # return tracker
    return None


def enhance_image(model, yolo_model, tracker, frame, counted_ids):

    original = frame.copy()

    img = cv2.resize(frame,(256,256))
    img = img/255.0
    img = np.transpose(img,(2,0,1))

    tensor = torch.tensor(img,dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    output = output.squeeze().cpu().numpy()
    output = np.transpose(output,(1,2,0))
    output = (output*255).astype(np.uint8)

    output = cv2.resize(output,(original.shape[1], original.shape[0]))

    # Blend original + enhanced
    output = cv2.addWeighted(original,0.6,output,0.4,0)

    # Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    output = cv2.filter2D(output,-1,kernel)

    # CLAHE contrast
    lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    output = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

    # YOLO Detection
    detected = output.copy()

    detections = []

    results = yolo_model.predict(
        source=output,
        conf=0.25,
        verbose=False
    )

    detections_for_tracker = []
    classes = []

    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            score = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            detections_for_tracker.append([x1,y1,x2,y2,score])
            classes.append(label)

            detections.append({
                "Object": label,
                "Confidence": round(score,2)
            })

    if len(detections_for_tracker) == 0:
        return original, output, detected, detections, counted_ids, 0, 0

    detections_for_tracker = np.array(detections_for_tracker)

    # SORT Tracking
    # tracks = tracker.update(detections_for_tracker)

    people_count = 0
    vehicle_count = 0

    for i, track in enumerate(tracks):
        x1,y1,x2,y2,track_id = map(int,track)
        track_id = int(track_id)

        cv2.rectangle(detected,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(
            detected,
            f"ID {track_id}",
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )

        if track_id not in counted_ids and i < len(classes):
            counted_ids.add(track_id)
            label = classes[i]

            if label == person_class:
                people_count += 1

            if label in vehicle_classes:
                vehicle_count += 1

    return original, output, detected, detections, counted_ids, people_count, vehicle_count
