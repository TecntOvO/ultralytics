from jinja2.optimizer import optimize

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from torch.nn.modules.conv import Conv2d



if __name__ == '__main__':
    model = YOLO(r"D:\Yolov11\ultralytics\runs\detect\train-test5-xs-0.001-nolayerdrop-lastpatch=1-bestbest\weights\best.pt")
    result = model.predict(source=r"D:\Yolov11\datasets\test_split_whole_7-2-1",save=True,save_score=True)
    # Dataset_ = YOLODataset(data="straberry_enh.yaml")