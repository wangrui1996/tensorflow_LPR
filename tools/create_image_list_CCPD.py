import os
import cv2
import sys


from tools.config import dataset, config, default,generate_config
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import json
import numpy as np
# set log level
tf.logging.set_verbosity(tf.logging.INFO)
generate_config(default.network, default.dataset)

thread_num = 100
provinces_num = {}
index = 1
lock = multiprocessing.Lock()


with open(config.char_map_json_file, "r") as f:
    char_label_map = json.load(f)
with open(config.province_map_json_file, "r") as f:
    province_label_map = json.load(f)

def find_key_by_value(label_map, value):
    for label in label_map:
        if label_map[str(label)] == value:
            return label

def write_to_files(dataset_path, image_save_path, image_list_path, dataset_name, executor, executor_transformation):
    img_txt_file = open(image_list_path, "w+")
    import time
    import random
    random.seed(time.time())
    for r, d, f in os.walk(dataset_path):
        for file in f:
            if ".jpg" in file:
                img_path = os.path.join(r, file)
                save_img_path = os.path.join(image_save_path, file)
                # - roi
                roi_str = img_path.split("-")[2]
                img_xy = img_path.split("-")[3]
                x1=int(roi_str.split('_')[0].split('&')[0])
                y1=int(roi_str.split('_')[0].split('&')[1])
                x2=int(roi_str.split('_')[1].split('&')[0])
                y2=int(roi_str.split('_')[1].split('&')[1])
                x11, y11 = int(img_xy.split("_")[0].split("&")[0]), int(img_xy.split("_")[0].split("&")[1])
                x01, y01 = int(img_xy.split("_")[1].split("&")[0]), int(img_xy.split("_")[1].split("&")[1])
                x00, y00 = int(img_xy.split("_")[2].split("&")[0]), int(img_xy.split("_")[2].split("&")[1])
                x10, y10 = int(img_xy.split("_")[3].split("&")[0]), int(img_xy.split("_")[3].split("&")[1])
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
                            sys.stdout.write(
                                '\r>>Writing to {:s} {:d}'.format(dataset_path, index))
                            sys.stdout.flush()


                executor.submit(progress, img_path, save_img_path, x1,y1,x2,y2)
                img_label = img_path.split("-")[4]
                img_label_list = [int(i) for i in img_label.split("_")]
                img_label = ""
                for i in img_label_list:
                    img_label = img_label + find_key_by_value(char_label_map, i)
                img_label = "{}{}".format(find_key_by_value(province_label_map, img_label_list[0]), img_label[1:])
                img_txt_file.writelines("{} {}\n".format(os.path.join(dataset_name, os.path.basename(save_img_path)), img_label.lower()))

                def progress_transform(image_save_path_, img_path_, pts1):
                    image = cv2.imread(img_path_)
                    pts2 = np.float32([[0, 0], [0, config.image_height], [config.image_width, 0], [config.image_width, config.image_height]])
                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    dst = cv2.warpPerspective(image, M, (config.image_width, config.image_height))
                    cv2.imwrite(image_save_path_, dst)
                    with lock:
                        global index
                        index = index + 1
                        if index % 1000 == 0:
                            sys.stdout.write(
                                '\r>>Writing to {:s} {:d}'.format(dataset_path, index))
                            sys.stdout.flush()
                crop_size = 8
                for idx in range(provinces_num[find_key_by_value(province_label_map, img_label_list[0])]):
                    img_save_base_name = "{}{}".format(idx, os.path.basename(save_img_path))
                    pts1 = np.float32([[x00 + random.randint(-crop_size, crop_size), y00 + random.randint(-crop_size, crop_size)],
                                       [x01 + random.randint(-crop_size, crop_size), y01 + random.randint(-crop_size, crop_size)],
                                       [x10 + random.randint(-crop_size, crop_size), y10 + random.randint(-crop_size, crop_size)],
                                       [x11 + random.randint(-crop_size, crop_size), y11 + random.randint(-crop_size, crop_size)]])
                    save_img_tmp_path = os.path.join(image_save_path, img_save_base_name)
                    executor_transformation.submit(progress_transform, save_img_tmp_path,img_path, pts1)
                    img_txt_file.writelines("{} {}\n".format(os.path.join(dataset_name, img_save_base_name), img_label.lower()))



def get_province_num(dataset_path):
    province_max = 0
    global provinces_num
    for r, d, f in os.walk(dataset_path):
        for file in f:
            if ".jpg" in file:
                img_path = os.path.join(r, file)
                # - roi
                img_label = img_path.split("-")[4]
                img_province_label = int(img_label.split("_")[0])
                img_province_key_label = find_key_by_value(province_label_map, img_province_label)
                value = provinces_num.get(img_province_key_label, 0)
                provinces_num[img_province_key_label] = value + 1
    for province_num in provinces_num:
        current_province_num = provinces_num[province_num]
        if current_province_num > province_max:
            province_max = current_province_num
    print(provinces_num)
    for province_num in provinces_num:
        current_province_num = provinces_num[province_num]
        provinces_num[province_num] = province_max // current_province_num
    print(provinces_num)




    #img_txt_file = open(image_list_path, "w+")



def make_image_list():


    plate_label_map_path = config.plate_map_json_file



    trainval_datasets = config.trainval_targets

    for dataset_name in trainval_datasets:
        global index
        with lock:
            index = 0
        executor = ThreadPoolExecutor(thread_num)
        executor_transformation = ThreadPoolExecutor(thread_num)
        tf.logging.info("Loading dataset: {} ".format(dataset_name))
        dataset_path = os.path.join(config.dataset_root_path, "ccpd_{}".format(dataset_name))
        image_save_root_path = os.path.join(config.data_store_path, "images")
        if not os.path.exists(image_save_root_path):
            os.mkdir(image_save_root_path)
        image_save_subdataset_path = os.path.join(image_save_root_path, dataset_name)
        if not os.path.exists(image_save_subdataset_path):
            os.mkdir(image_save_subdataset_path)
        image_list_path = os.path.join(config.data_store_path, "{}.txt".format(dataset_name))
        get_province_num(dataset_path)
        write_to_files(dataset_path, image_save_subdataset_path, image_list_path, dataset_name, executor, executor_transformation)
        executor.shutdown(wait=True)
        executor_transformation.shutdown(wait=True)
        print("\n")
        with lock:
            tf.logging.info("Top handle {} in {}".format(index, dataset_name))


    for key in province_label_map:
        province_label_map[key] = province_label_map[key] + len(char_label_map)
    with open(plate_label_map_path, "w+") as f:
        json.dump({**char_label_map, **province_label_map}, f)

    print("finished")

def main():
    make_image_list()

if __name__ == '__main__':
    main()
