from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO(r"D:\USTH\Computer_Vision\table\model\table_detect_model.pt")
# hoặc đường dẫn khác tuỳ bạn
table_id = 337
img_path = fr"D:\USTH\Computer_Vision\table\crop_table\images\test\table_{table_id}.png"
img_path = r"D:\USTH\Computer_Vision\table\dataset\images\test\ZBRA_2017_page_79.png"
results = model.predict(
    source=img_path,
    conf=0.01,     # threshold confidence
    iou=0.5,
    save=False,
    verbose=False
)
result = results[0]

# ảnh gốc (BGR)
img = result.orig_img.copy()

# bbox: xyxy, conf, class
boxes = result.boxes

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls = int(box.cls[0])
    label = f"{model.names[cls]} {conf:.2f}"

    # vẽ bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2, 1)
    cv2.putText(
        img,
        f"{conf:.2f}",
        (x1, y1 - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,          # chữ bé
        (0, 255, 0),
        1            # thickness chữ
    )
    
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()