import os
import cv2
from tools.config import dataset
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import json

thread_num = 100

image_list_path = dataset.image_list_path
dataset_path = dataset.dataset_path
data_save_path = dataset.data_save_path

char_label_map_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "char_map/char_map_CCPD.json")
with open(char_label_map_path, "r") as f:
    char_label_map = json.load(f)

province_label_map_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "char_map/province_map_CCPD.json")
with open(province_label_map_path, "r") as f:
    province_label_map = json.load(f)

plate_label_map_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "char_map/plate_map.json")




def find_key_by_value(label_map, value):
    for label in label_map:
        if label_map[str(label)] == value:
            return label

img_txt_file = open(image_list_path, "w+")

plate_list = []

index = 1
lock = multiprocessing.Lock()
executor = ThreadPoolExecutor(thread_num)
for r, d, f in os.walk(dataset_path):
    for file in f:
        if ".jpg" in file:
            img_path = os.path.join(r, file)
            save_img_path = os.path.join(data_save_path, file)
            # - roi
            roi_str = img_path.split("-")[2]
            x1=int(roi_str.split('_')[0].split('&')[0])
            y1=int(roi_str.split('_')[0].split('&')[1])
            x2=int(roi_str.split('_')[1].split('&')[0])
            y2=int(roi_str.split('_')[1].split('&')[1])
            def progress(image_path, save_image_path, x1, y1, x2, y2):
                try:
                    img = cv2.imread(image_path)
                except:
                    print("can not read '{}'".format(image_path))

                img = img[y1:y2, x1:x2, :]
                cv2.imwrite(save_image_path, img)
                with lock:
                    global index
                    index = index + 1
                    if index % 1000 == 0:
                        print("pregress {}".format(index))


            executor.submit(progress, img_path, save_img_path, x1,y1,x2,y2)
            img_label = img_path.split("-")[4]
            img_label_list = [int(i) for i in img_label.split("_")]
            img_label = ""
            for i in img_label_list:
                img_label = img_label + find_key_by_value(char_label_map, i)
            img_label = "{}{}".format(find_key_by_value(province_label_map, img_label_list[0]), img_label[1:])
            img_txt_file.writelines("{} {}\n".format(os.path.basename(save_img_path), img_label.lower()))
executor.shutdown(wait=True)
for key in province_label_map:
    province_label_map[key] = province_label_map[key] + len(char_label_map)


with open(plate_label_map_path, "w+") as f:
    json.dump({**char_label_map ,**province_label_map}, f)

print("\n")
