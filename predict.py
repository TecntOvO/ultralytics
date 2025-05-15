from jinja2.optimizer import optimize

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from torch.nn.modules.conv import Conv2d



if __name__ == '__main__':
    model = YOLO(r"D:\Yolov11\ultralytics\runs\detect\循序渐进改进\Baseline\weights\best.pt")
    # model = YOLO( r"D:\Yolov11\ultralytics\runs\detect\train-yolon\weights\best.pt")
    model.predict(source=r"D:\Yolov11\datasets\split_final\images\val",save=True,save_score=False,save_txt=False, batch=1, iou=0.5, line_width=2)
    # model.predict(source=r"D:\Yolov11\datasets\test_split_whole_7-2-1", save=True, show_labels=False, show_conf=False, iou=0.0, conf=0.01, max_det=5000)
    # Dataset_ = YOLODataset(data="straberry_enh.yaml")