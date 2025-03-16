from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\Yolov11\intense_detect\ultralytics\runs\detect\yolov+SIOU(640)+InnerIoU1.20+Attentionv2\weights\best.pt')
    model.predict(
        # source=r"D:\Yolov11\intense_detect\MOT20-01\data_1_enhance\images\test",
        source=r"D:\Yolov11\intense_detect\MOT20-01\data_1_enhance\images\eval",
        save=True,
        # show_conf=False,
        # show_labels=False,
        # save_txt=False,
        line_width=3,
        save_score=True,
        iou=0.5,
        imgsz=640,
        # score_threshold=0.5
    )
    model1 = YOLO(r'D:\Yolov11\intense_detect\ultralytics\runs\detect\yolov+SIOU(640)+InnerIoU1.20+Attentionv2\weights\best.pt')
    model1.predict(
        source=r"D:\Yolov11\intense_detect\MOT20-01\07_eval",
        # source=r"D:\Yolov11\intense_detect\MOT20-01\07",
        save=True,
        # show_conf=False,
        # show_labels=False,
        # save_txt=False,
        line_width=3,
        save_score=True,
        iou=0.5,
        imgsz=640,
        # score_threshold=0.5
    )