from ultralytics import YOLO

# Load model đã train
model = YOLO(r"D:\USTH\Computer_Vision\table\model\table_detect_model.pt")   # hoặc yolov8n.pt nếu test pretrained

# Đánh giá trên tập test


#model.predict(
    #source=r"D:\USTH\Computer_Vision\table\dataset\images\test",
    #save=True,
    #conf=0.1,
    #imgsz=640)

metrics = model.val(
    data=r"D:\USTH\Computer_Vision\table\dataset\data.yaml",
    split="test",     # dùng test set
    imgsz=(608, 960),
    conf=0.1,
    iou=0.5,     
)

# In kết quả chính
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
