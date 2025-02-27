from jinja2.optimizer import optimize

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from torch.nn.modules.conv import Conv2d



if __name__ == '__main__':
    # model = YOLO(r"yolo11+ViT.yaml")
    # model = YOLO(r"yolov11-ViTv2.yaml")
    # model = YOLO(r"yolo11n.yaml")
    # model = YOLO(r"yolo11s.yaml")
    # model = YOLO(r"yolov11-ViTv2-NewNeck-test4-xxs.yaml").load(
    #     r"D:\Yolov11\ultralytics\runs\detect\compare in newNeck\new new data\mv2-cov24-xxs-0.001-ffndrop0.0 allmv2\weights\best.pt")
    model = YOLO(r"yoloA11.yaml")
    # model.load(r"yolo11n.pt")
    model.train(
        # data="straberry_enh.yaml",  # path to dataset YAML
        # data="straberry.yaml",
        data="straberry_enhance.yaml",
        epochs=200,  # number of training epochs
        lr0=0.001,
        cfg="ours.yaml",
        rect=True,
        batch=32,
        resume=False
    )
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
