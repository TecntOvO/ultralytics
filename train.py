from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.yaml')
    model.load("yolov8n.pt")
    model.train(
        data="mat20-01_enhance.yaml",
        epochs=150,  # number of training epochs
        lr0=0.001,
        lrf=0.01,
        rect=True,
        batch=8,
        workers=8,
        show_conf=False,
        show_labels=False,
        line_width=1,
        patience=50,
        imgsz=1024,
        weight_decay=0.0007,
        val_coco_path=r"K:\Yolov8\work\MOT20-01\data_1_enhance\val.json",
        test_coco_path=r"K:\Yolov8\work\MOT20-01\data_1_enhance\test.json",
    )