from jinja2.optimizer import optimize

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from torch.nn.modules.conv import Conv2d



if __name__ == '__main__':
    # model = YOLO(r"yolo11+ViT.yaml")
    # model = YOLO(r"yolov11-ViTv2.yaml")
    model = YOLO(r"yolo11n.yaml")
    # model = YOLO(r"yolo11s.yaml")
    train_results = model.train(
        # data="straberry_enh.yaml",  # path to dataset YAML
        # data="straberry.yaml",
        data="straberry_.yaml",
        epochs=300,  # number of training epochs
        lr0=0.01,
        cfg="ours.yaml",
        rect=True,
        batch=32
    )
    # Dataset_ = YOLODataset(data="straberry_enh.yaml")