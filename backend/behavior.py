import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# 1. Загрузка модели (YOLOv8n - самая быстрая для тестов)
model = YOLO('yolov8n.pt') # Для коров/овец лучше дообучить на кастомных данных

# Словарь для хранения истории перемещений {track_id: [координаты]}
track_history = defaultdict(lambda: [])
# Словарь для хранения общей пройденной дистанции
distance_data = defaultdict(float)

# Открываем видео (0 для веб-камеры или путь к файлу)
video_path = "pasture_video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 2. Детекция и Трекинг (persist=True сохраняет ID между кадрами)
    # classes=[19, 20, 21] — это коровы, лошади, овцы в датасете COCO
    results = model.track(frame, persist=True, classes=[19, 20, 21], verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            current_pos = np.array([float(x), float(y)])
            
            # 3. Расчет дистанции
            if track_id in track_history:
                prev_pos = track_history[track_id][-1]
                dist = np.linalg.norm(current_pos - prev_pos)
                
                # Фильтруем шум (если движение меньше 2 пикселей, не считаем)
                if dist > 2:
                    distance_data[track_id] += dist
            
            # Сохраняем историю
            track_history[track_id].append(current_pos)
            if len(track_history[track_id]) > 30: # Храним только последние 30 точек
                track_history[track_id].pop(0)

            # 4. Логика аномалий (пример)
            total_dist = distance_data[track_id]
            status = "Active"
            color = (0, 255, 0) # Зеленый
            
            if total_dist < 50: # Порог подбирается эмпирически
                status = "LOW ACTIVITY!"
                color = (0, 0, 255) # Красный

            # Отрисовка
            cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 2)
            cv2.putText(frame, f"ID: {track_id} | {status}", (int(x-w/2), int(y-h/2)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Livestock Health Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()