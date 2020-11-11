import torch
from draw_box_utils import draw_box
from PIL import Image
import json
import matplotlib.pyplot as plt
from train_ssd512 import create_model
import transform
import time
import cv2
import numpy as np

# get devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# create model
model = create_model(num_classes=3)

# load train weights
train_weights = "./save_weights/ssd512-24.pth"
train_weights_dict = torch.load(train_weights, map_location=device)['model']

model.load_state_dict(train_weights_dict, strict=False)
model.to(device)

# read class_indict
category_index = {}
try:
    json_file = open('./night_classes.json', 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

val_path = r'F:\JS\SSD_ResNet\NightData\ImageSets\Main\val.txt'
with open(val_path) as read:
    num_ids = [line.strip() for line in read.readlines()]

for num in num_ids:
    # load image
    original_img = Image.open("F:/JS/SSD_ResNet/NightData/JPEGImages/sample" + str(num) + ".jpg")

    # from pil image to tensor, do not normalize image
    data_transform = transform.Compose([transform.Resize(),
                                        transform.ToTensor(),
                                        transform.Normalization()])
    img, _ = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        # starttime = time.time()
        predictions = model(img.to(device))[0]  # bboxes_out, labels_out, scores_out
        # endtime = time.time()
        # dtime = endtime - starttime
        # print(dtime)

        predict_boxes = predictions[0].to("cpu").numpy()
        predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * original_img.size[0]
        predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * original_img.size[1]
        predict_classes = predictions[1].to("cpu").numpy()
        predict_scores = predictions[2].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        draw_box(original_img,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 category_index,
                 thresh=0.05,
                 line_thickness=2)
        # plt.imshow(original_img)
        # plt.show()

    img = cv2.cvtColor(np.asarray(original_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite("./save_imgs/sample" + str(num) + ".jpg", img)

    # cv2.waitKey(0)


