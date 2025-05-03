from ultralytics import YOLO
import os

if __name__ == '__main__':
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

    model = YOLO(r"yolo11-11-base-P5.yaml")
    # mask = {13: 11, 16: 14, 17: 15, 19: 17, 20:18, 22:20, 23:21}
    mask = {16:9, 17:10, 19:12, 23:13}
    # model.load(r"yolo11n.pt", mask)
    model.load(r"D:\Yolov11\ultralytics\runs\detect\循序渐进改进\Baseline\weights\last.pt", mask)
    model.train(
        warmup_epochs=0,
        data="straberry_final_enhance.yaml",
        epochs=100,  # number of training epochs
        cfg='ours.yaml',
        lr0=0.0001,
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
    # model.train(
    #     data="coco.yaml",
    #     epochs=10,  # number of training epochs
    #     imgsz=640
    # )
