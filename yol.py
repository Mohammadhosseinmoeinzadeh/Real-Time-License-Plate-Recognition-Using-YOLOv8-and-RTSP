from ultralytics import YOLO
from pathlib import Path

data_yaml = """
path: <YOR_PATH_DATASET>
train: <YOUR_train_ADDRESS>
val: <YOUR_val_ADDRESS>

nc: 1
names: ['plate']

"""

with open("data.yaml", "w") as f:
    f.write(data_yaml)


model = YOLO("yolov8n.pt")  


model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="plate-detector"
)


results = model("test_image.jpg")  

results[0].show()


results[0].save(filename="output.jpg")

