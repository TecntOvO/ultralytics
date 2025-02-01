from jinja2.optimizer import optimize

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from torch.nn.modules.conv import Conv2d



if __name__ == '__main__':
    model = YOLO(r"D:\Yolov11\ultralytics\runs\detect\train vitv2-s+test2 new- 350e\weights\best.pt")
    result = model.predict(source=r"D:\Yolov11\datasets\test",save=True)
    # Dataset_ = YOLODataset(data="straberry_enh.yaml")