# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# import cv2
# import numpy as np
# import io
# import math
# from typing import List
# import time
# from model_class import model, classNames


# def load_image(file: UploadFile) -> np.ndarray:
#     image_bytes = file.read()
#     nparr = np.fromstring(image_bytes, np.uint8)
#     return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# def run_model(image: np.ndarray) -> List[dict]:
#     results = model(image, stream=True)
#     detected_classes = []
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             cls = int(box.cls[0])
#             if cls > len(classNames)-1:
#                 continue
#             class_name = classNames[cls]
#             detected_classes.append({"class_name": class_name, "confidence": conf, "bbox": (x1, y1, x2, y2)})
#     return detected_classes

# def draw_bounding_boxes(image: np.ndarray, detected_classes: List[dict]) -> np.ndarray:
#     for detection in detected_classes:
#         class_name, conf = detection["class_name"], detection["confidence"]
#         label = f'{class_name} {conf}'
#         x1, y1, x2, y2 = [int(x) for x in detection["bbox"]]
#         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
#         t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#         c2 = x1 + t_size[0], y1 - t_size[1] - 3
#         cv2.rectangle(image, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
#         cv2.putText(image, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
#     return image

import cv2
import numpy as np
import io
import math
from model_class import model, classNames

def process_image(image_bytes):
    # Decode the image
    nparr = np.fromstring(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run the model on the image
    results = model(img, stream=True)
    
    detected_classes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if cls > len(classNames)-1:
                continue
            class_name = classNames[cls]
            label = f'{class_name} {conf}'
            detected_classes.append({"class_name": class_name, "confidence": conf})
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
    # Encode the processed image to JPEG format
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = io.BytesIO(img_encoded.tobytes())
    
    return img_bytes, detected_classes

def process_video_stream():
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    while True:
        success, img = cap.read()
        if not success:
            break
        
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if cls > len(classNames)-1:
                    continue
                class_name = classNames[cls]
                label = f'{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def process_detected_classes_stream():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break
        
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls > len(classNames)-1:
                    continue
                detected_class = classNames[cls]
                yield f"data: {detected_class}\n\n"
    cap.release()
