from ultralytics import YOLO
import os

if __name__ == '__main__':
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

    model = YOLO(r"yolo11A11-9.yaml")
    # model.load(r"yolo11n.pt")
    model.load_(r"mobilevitv2-0.5.pt")
    model.train(
        # data="straberry.yaml",
        data="straberry__enhance.yaml",
        epochs=120,  # number of training epochs
        cfg='ours.yaml',
        lr0=0.0001,
        lrf=0.01,
        rect=True,
        batch=32,
        resume=True,
        cos_lr=True,
        export_coco_result=False,
        weight_decay=0.0007,
        patience=50,
        val_coco=r"D:\Yolov11\datasets\split_whole_enhance_8-2_new\instances_val2017.json",
        iou_type='MPDiou',
        Inner_iou=True,
        ratio=0.85,
        workers=8
    )
