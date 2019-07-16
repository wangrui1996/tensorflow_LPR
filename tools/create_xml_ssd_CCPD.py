import os
import cv2
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import random
import xml.etree.ElementTree as ET
dataset_path = "/home/rui/data/CCPD2019/ccpd_base"
thread_num = 8
lock = multiprocessing.Lock()
executor = ThreadPoolExecutor(thread_num)
annotation_names = []
xml_save_path = ""


for r, d, f in os.walk(dataset_path):
    for file in f:
        if ".jpg" in file:
            img_path = os.path.join(r, file)
            # - roi
            roi_str = img_path.split("-")[2]
            x1=int(roi_str.split('_')[0].split('&')[0])
            y1=int(roi_str.split('_')[0].split('&')[1])
            x2=int(roi_str.split('_')[1].split('&')[0])
            y2=int(roi_str.split('_')[1].split('&')[1])

            save_img_path = os.path.join(xml_save_path, "{}.xml".format(file.split(".")[0]))

            def progress(image_path, save_xml_path, x1, y1, x2, y2):
                try:
                    img = cv2.imread(image_path)
                    h,w,c = img.shape
                    annotation = ET.Element("annotation")
                    size = ET.SubElement(annotation, "size")
                    ET.SubElement(size, "width").text = str(w)
                    ET.SubElement(size, "height").text = str(h)
                    object = ET.SubElement(annotation, "object")
                    ET.SubElement(object, "name").text = "plate"
                    ET.SubElement(object, "difficult").text = "0"
                    bndbox = ET.SubElement(object, "bndbox")
                    ET.SubElement(bndbox, "xmin").text = str(x1)
                    ET.SubElement(bndbox, "ymin").text = str(y1)
                    ET.SubElement(bndbox, "xmax").text = str(x2)
                    ET.SubElement(bndbox, "ymax").text = str(y2)
                    tree = ET.ElementTree(annotation)
                    tree.write(save_xml_path)
                except:
                    print("can not read '{}'".format(image_path))

                with lock:
                    global annotation_names
                    annotation_names.append(image_path.split(".")[0])
                    global index
                    index = index + 1
                    if index % 1000 == 0:
                        print("pregress {}".format(index))


            executor.submit(progress, img_path, save_img_path, x1,y1,x2,y2)

executor.shutdown(wait=True)



random.shuffle(annotation_names)

anchor = int(len(annotation_names)*0.9)

train = annotation_names[:anchor]
test = annotation_names[anchor:]

with open("test.txt", "w+") as f:
    for name in test:
        f.write("{}\n".format(name))

with open("trainval.txt", "w+") as f:
    for name in train:
        f.write("{}\n".format(name))
