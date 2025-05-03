from ultralytics import YOLO
import os

if __name__ == '__main__':
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    model = YOLO(r"yolo11-compare2.yaml")
    # mask = {13: 11, 16: 14, 17: 15, 19: 17, 20:18, 22:20, 23:21}
    mask = {13: 11, 16: 14, 17: 15, 19: 17, 23: 18}
    # mask = {16: 9, 17: 10, 19: 12, 23: 13}
    # model.load(r"D:\Yolov11\ultralytics\runs\detect\循序渐进改进\+2 C3k2A 不同layer对比\2 - 2 p=2 - 2 best\weights\best.pt", {21: 18})
    model.load(r"yolo11n.pt", mask)

    # model.load(r"D:\Yolov11\ultralytics\runs\detect\循序渐进改进\-P5 + 2 C3k2A  2\weights\best.pt")
    model.train(
        data="straberry_final_enhance.yaml",
        epochs=150,  # number of train ing epochs
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
        iou_type='MPDiou',
        Inner_iou=False,
        ratio=0.85,
        workers=8,
        # freeze=[0,1,2,3,4,5,6,7,8],
        # warmup_epochs=0.0,
    )


    # print('####################################################################')
    # model = YOLO(r"yoloA11-2.yaml")
    # model.load(r"yolo11n.pt")
    # model.train(
    #     # data="straberry_enh.yaml",  # path to dataset YAML
    #     # data="straberry.yaml",
    #     data="straberry_.yaml",
    #     epochs=1,  # number of training epochs
    #     lr0=0.001,
    #     cfg="ours.yaml",
    #     rect=True,
    #     batch=32,
    #     resume=False
    # )
    # print('####################################################################')
    # model = YOLO(r"yolo11n.pt")
    # model.train(
    #     # data="straberry_enh.yaml",  # path to dataset YAML
    #     # data="straberry.yaml",
    #     data="straberry_.yaml",
    #     epochs=150,  # number of training epochs
    #     lr0=0.001,
    #     cfg="ours.yaml",
    #     rect=True,
    #     batch=32
    # )
    # print('####################################################################')
    # model = YOLO(r"yoloA11-2.yaml")
    # model.load(r"yolo11n.pt")
    # model.train(
    #     # data="straberry_enh.yaml",  # path to dataset YAML
    #     # data="straberry.yaml",
    #     data="straberry_.yaml",
    #     epochs=150,  # number of training epochs
    #     lr0=0.001,
    #     cfg="ours.yaml",
    #     rect=True,
    #     batch=32
    # )
    # print('####################################################################')
    # model = YOLO(r"yoloA11-3.yaml")
    # model.load(r"yolo11n.pt")
    # model.train(
    #     # data="straberry_enh.yaml",  # path to dataset YAML
    #     # data="straberry.yaml",
    #     data="straberry_.yaml",
    #     epochs=150,  # number of training epochs
    #     lr0=0.001,
    #     cfg="ours.yaml",
    #     rect=True,
    #     batch=32
    # )
    # model.train(
    #     # data="straberry_enh.yaml",  # path to dataset YAML
    #     # data="straberry.yaml",
    #     data="straberry_.yaml",
    #     epochs=250,  # number of training epochs
    #     lr0=0.001,
    #     cfg="ours.yaml",
    #     rect=True,
    #     batch=32
    # )
