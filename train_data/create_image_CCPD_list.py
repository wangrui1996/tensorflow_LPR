import os
import cv2
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

thread_num = 100
image_list_path = "/home/rui/work/crnn_ctc_ocr.Tensorflow/train_data/image_list.txt"
dataset_path = "/home/rui/data/CCPD2019/ccpd_base"
data_save_path = "/home/rui/work/crnn_ctc_ocr.Tensorflow/data"

label_map = ["A","B","C","D","E","F" \
        "G","H","I","J","K","L","M","N","O","P",\
        "Q","R","S","T","U","V","W","X","Y","Z",\
        "0","1","2","3","4","5","6","7","8","9"]

img_txt_file = open(image_list_path, "w+")

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
            x1=int(roi_str.split("_")[0].split("&")[0])
            y1=int(roi_str.split("_")[0].split("&")[1])
            x2=int(roi_str.split("_")[1].split("&")[0])
            y2=int(roi_str.split("_")[1].split("&")[1])
            def progress(image_path, save_image_path, x1, y1, x2, y2):
                img = cv2.imread(image_path)
                img = img[y1:y2, x1:x2, :]
                cv2.imwrite(save_image_path, img)
                with lock:
                    global index
                    index = index + 1
                    if index % 1000 == 0:
                        print("pregress {}".format(index))


            executor.submit(progress, img_path, save_img_path, x1,y1,x2,y2)
            #progress(img_path, save_img_path, x1,y1,x2,y2)
            #img = cv2.imread(img_path)
            #img = img[y1:y2, x1:x2,:]
            #cv2.imwrite(save_img_path, img)
            #print(roi_str)
            #print(x1, y1, x2, y2)
            img_label = img_path.split("-")[4]
            img_label_list = [int(i) for i in img_label.split("_")]
            #print(img_label_list)
            #exit(0)
            img_label = ""
            for i in img_label_list:
                img_label = img_label + label_map[i] 
            img_label = img_label[1:] 
            img_txt_file.writelines("{} {}\n".format(save_img_path, img_label.lower()))
            #exit(0)
executor.shutdown(wait=True)
print("\n")
