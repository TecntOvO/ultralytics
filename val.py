from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO(r"D:\Yolov11\ultralytics\runs\detect\impove on yolo\yolon-hsvreset\weights\best.pt")
    result = model.val(data="straberry_.yaml", export_coco_result=True, val_coco="D:\Yolov11\datasets\split_whole_7-3\instances_val2017.json")
    # Dataset_ = YOLODataset(data="straberry_enh.yaml")