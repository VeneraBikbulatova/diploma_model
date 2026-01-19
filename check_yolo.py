from ultralytics import YOLO
import cv2

# Загрузка обученной модели
model = YOLO("runs/detect/violin_detector/weights/best.pt")

# Предсказание на новых данных
results = model.predict(
    source="dataset/images/val",
    conf=0.5,  # Порог уверенности
    save=True,  # Сохранить результаты
    save_txt=True,  # Сохранить метки
    save_conf=True  # Сохранить уверенность
)

# Визуализация
for result in results:
    img = result.plot()  # Рисует bbox на изображении
    cv2.imshow("Detection", img)
    cv2.waitKey(0)
    
model = YOLO("runs/detect/violin_detector/weights/best.pt")
metrics = model.val()  # Автоматически использует данные из data.yaml

print(f"mAP50-95: {metrics.box.map}")  # Основная метрика
print(f"Class-specific AP:")
for i, name in enumerate(model.names):
    print(f"{name}: {metrics.box.map50[i]:.3f}")