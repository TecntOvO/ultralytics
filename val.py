from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO(r"D:\Yolov11\ultralytics\runs\detect\循序渐进改进\YOLOv11 m\weights\best.pt")
    result = model.val(data="straberry_final_enhance.yaml", cfg='ours.yaml')
    # Dataset_ = YOLODataset(data="straberry_enh.yaml")