from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'K:\Yolov8\ultralytics\runs\detect\train59\weights\best.pt')
    model.val(
        data="mat20-01_enhance.yaml",
        rect=True,
        batch=8,
        show_conf=False,
        show_labels=False,
        line_width=1,
        workers=1,
        val_coco_path=r"K:\Yolov8\work\MOT20-01\data_1_enhance\val.json"
    )