from ultralytics import YOLO

model = YOLO(r"D:\turn-detection-project\model\weights.pt")
print("模型載入成功")
print("類別名稱：", model.names)
