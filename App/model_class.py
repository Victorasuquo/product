from ultralytics import YOLO
model = YOLO("Model/best.pt")
classNames = ['Eye_Closed', 'Eye_Open', 'Facing_Front', 'Mouth_Yawning']
print("... Model Loaded Successfully | class initialized")