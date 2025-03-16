from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\Yolov11\intense_detect\ultralytics\runs\detect\yolov+Attentionv2(p=1,e=1.0)(640）\weights\best.pt')
    model.val(
        data="mat20-01_enhance.yaml",
        rect=True,
        batch=8,
        show_conf=False,
        show_labels=False,
        line_width=3,
        imgsz=640,
        workers=1,
        plots=False
    )
    model.val(
        data="mat20-01_enhance_.yaml",
        rect=True,
        batch=8,
        show_conf=False,
        show_labels=False,
        line_width=3,
        imgsz=640,
        workers=1,
        plots=False
    )