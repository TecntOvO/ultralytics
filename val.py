from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO(r"D:\Yolov11\ultralytics\runs\detect\train5\weights\best.pt")
    result = model.val(data="straberry__.yaml",verbose=True)
    # Dataset_ = YOLODataset(data="straberry_enh.yaml")