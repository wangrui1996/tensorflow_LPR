import os
import cv2
from tools.config import dataset, config, default,generate_config
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import json
# set log level
tf.logging.set_verbosity(tf.logging.INFO)
generate_config(default.network, default.dataset)

thread_num = 100

index = 1
lock = multiprocessing.Lock()
executor = ThreadPoolExecutor(thread_num)

with open(config.char_map_json_file, "r") as f:
    char_label_map = json.load(f)
with open(config.province_map_json_file, "r") as f:
    province_label_map = json.load(f)

def find_key_by_value(label_map, value):
    for label in label_map:
        if label_map[str(label)] == value:
            return label

def write_to_files(dataset_path, image_save_path, image_list_path):
    img_txt_file = open(image_list_path, "w+")
    for r, d, f in os.walk(dataset_path):
        for file in f:
            if ".jpg" in file:
                img_path = os.path.join(r, file)
                save_img_path = os.path.join(image_save_path, file)
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
                            tf.logging.info("pregress {}".format(index))


                executor.submit(progress, img_path, save_img_path, x1,y1,x2,y2)
                img_label = img_path.split("-")[4]
                img_label_list = [int(i) for i in img_label.split("_")]
                img_label = ""
                for i in img_label_list:
                    img_label = img_label + find_key_by_value(char_label_map, i)
                img_label = "{}{}".format(find_key_by_value(province_label_map, img_label_list[0]), img_label[1:])
                img_txt_file.writelines("{} {}\n".format(os.path.basename(save_img_path), img_label.lower()))


#img_txt_file = open(image_list_path, "w+")



def make_image_list():


    plate_label_map_path = config.plate_map_json_file



    trainval_datasets = config.trainval_targets

    for dataset_name in trainval_datasets:
        tf.logging.info("Loading dataset: {} ".format(dataset_name))
        dataset_path = os.path.join(config.dataset_root_path, "ccpd_{}".format(dataset_name))
        image_save_root_path = os.path.join(config.data_store_path, "images")
        if not os.path.exists(image_save_root_path):
            os.mkdir(image_save_root_path)
        image_save_subdataset_path = os.path.join(image_save_root_path, dataset_name)
        if not os.path.exists(image_save_subdataset_path):
            os.mkdir(image_save_subdataset_path)
        image_list_path = os.path.join(config.data_store_path, "{}.txt".format(dataset_name))
        global index
        index = 0
        write_to_files(dataset_path, image_save_subdataset_path, image_list_path)

        tf.logging.info("\rFinished ...".format(dataset_name))
    executor.shutdown(wait=True)

    for key in province_label_map:
        province_label_map[key] = province_label_map[key] + len(char_label_map)
    with open(plate_label_map_path, "w+") as f:
        json.dump({**char_label_map, **province_label_map}, f)

    print("finished")

def main():
    make_image_list()

if __name__ == '__main__':
    main()
