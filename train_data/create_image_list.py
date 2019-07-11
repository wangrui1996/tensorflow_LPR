import os
image_list_path = "./image_list.txt"
dataset_path = "/home/rui/work/crnn_ctc_ocr.Tensorflow/90kDICT32px"


img_txt_file = open(image_list_path, "w+")

index = 1

for r, d, f in os.walk(dataset_path):
    for file in f:
        if ".jpg" in file:
            img_path = os.path.join(r, file)
            img_label = img_path.split("/")[-1].split(".jpg")[0].split("_")[1]
            img_txt_file.writelines("{} {}\n".format(img_path, img_label.lower()))
        if index %10000 == 0:
            print("pregress {}".format(index))
            exit(0)

        index = index + 1
print("\n")
