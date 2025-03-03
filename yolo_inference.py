from ultralytics import YOLO

def load_model(model_name="yolov8n.pt"):
    model = YOLO(model_name)
    return model

def run_inference(model, frame):
    results = model(frame)
    return results