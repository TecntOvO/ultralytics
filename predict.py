from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'K:\Yolov8\ultralytics\runs\detect\train60\weights\best.pt')
    model.predict(
        # source=r"K:\Yolov8\work\MOT20-01\data_1_enhance\images\test",
        source=r"K:\Yolov8\work\MOT20-01\07",
        save=True,
        show_conf=False,
        show_labels=False,
        save_txt=False,
        line_width=1,
        save_score=False,
        imgsz=1024,
        # iou=0.4,
        # score_threshold=0.5
    )