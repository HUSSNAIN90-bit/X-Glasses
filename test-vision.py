from app.services.vision import detect_objects


detections = detect_objects("test.jpg")

for detection in detections:
    print(detection)