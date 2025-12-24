from ultralytics import YOLO

DATASET = r"D:\9.ATC_AI_Core\SVSC2026\DATASET\Animal3_yolo_dataset"


def train_yolo():
    model = YOLO(r"..\..\PreTrained\Yolo11_Classification\yolo11n-cls.pt")

    results = model.train(
        data=DATASET,
        epochs=10,
        imgsz=224,
        batch=256,
        device="cpu",
        workers=8,
        name="yolo11_cls_dog_cat_fox",
        plots=True,
        augment=True,
    )

    print("Training completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    train_yolo()
