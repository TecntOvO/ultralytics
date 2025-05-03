from ultralytics import YOLO
import os

if __name__ == '__main__':
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

    model = YOLO(r"yolo11-11-3.yaml")
    mask = {16:9, 17:10, 23:13, 19:12}
    # mask = {23:13}
    # mask = {13: 11, 16: 14, 17: 15, 19: 17, 20: 18, 22: 20, 23: 21}
    model.load(r"yolo11n.pt", mask)
    # mask2 = {6:-2}
    # freeze_list = [0,1,2,3,4,5,6]
    freeze_list = []
    # model.load(r"D:\Yolov11\ultralytics\runs\detect\final\yolon+RemoveP5+C3k2A+channel_attn4 embed = half c_in 0.0007\weights\best.pt")

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
        freeze=freeze_list
    )

    # model.train(
    #     # data="straberry.yaml",
    #     data="straberry_final_enhance.yaml",
    #     epochs=150,  # number of training epochs
    #     cfg='ours.yaml',
    #     lr0=0.0005,
    #     lrf=0.01,
    #     rect=True,
    #     batch=32,
    #     resume=False,
    #     cos_lr=True,
    #     export_coco_result=False,
    #     weight_decay=0.0007,
    #     patience=50,
    #     val_coco=r"D:\Yolov11\datasets\split_whole_enhance_8-2_new\instances_val2017.json",
    #     iou_type='MPDiou',
    #     Inner_iou=False,
    #     ratio=0.85,
    #     workers=8,
    #     freeze=freeze_list
    # )
