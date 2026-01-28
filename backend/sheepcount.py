import cv2
from ultralytics import solutions


def count_sheep_in_video(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Ошибка открытия видео")

    # Линия подсчёта (подстрой под своё видео!)
    region_points = [(200, 400), (1000, 400)]

    counter = solutions.ObjectCounter(
        show=True,             # Не показываем окно (важно для сервера)
        region=region_points,
        model="yolov8n.pt",
        classes=[18],           # 18 = sheep
    )

    while True:
        success, frame = cap.read()
        if not success:
            break

        counter(frame)

    cap.release()

    # Сколько овец вошло
    total_sheep = counter.in_count
    print(total_sheep)
    return total_sheep
