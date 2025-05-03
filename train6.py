from ultralytics import YOLO
import os

if __name__ == '__main__':
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

    model = YOLO(r"yolo11-11-3-1.yaml")
    mask = {16: 9, 17: 16, 23: 19, 19: 18}
    model.load(r"yolo11n.pt", mask)
    model.train(
        data="straberry_final_enhance.yaml",
        epochs=150,  # number of training epochs
        cfg='ours.yaml',
        lr0=0.0007,
        lrf=0.01,
        rect=True,
        batch=32,
        resume=False,
        cos_lr=True,
        export_coco_result=False,
        weight_decay=0.0007,
        patience=50,
        val_coco=r"D:\Yolov11\datasets\split_whole_enhance_8-2_new\instances_val2017.json",
        iou_type='SIoU',
        Inner_iou=False,
        ratio=0.85,
        workers=8,
    )
