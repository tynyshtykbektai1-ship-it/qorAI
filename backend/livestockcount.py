from ultralytics import YOLO

class AnimalCounter:
    def __init__(self, model_path='models/yolov8l.pt', classes=['sheep','cow'], conf=0.5):
     
        self.model = YOLO(model_path)
        self.classes = classes
        self.conf = conf

    def count(self, video_path, save=False):

        unique_ids_per_class = {cls: set() for cls in self.classes}

        # трекинг животных в видео
        for r in self.model.track(source=video_path, stream=True, conf=self.conf, save=save):
            for box in r.boxes:
                if box.id is None:
                    continue
                class_name = r.names[int(box.cls)]
                if class_name in self.classes:
                    unique_ids_per_class[class_name].add(int(box.id))

        counts = {cls: len(ids) for cls, ids in unique_ids_per_class.items()}
        return counts