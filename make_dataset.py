import os
import numpy as np
import random
from PIL import Image

path_name = "Dataset/"

train = open("train.txt","w+")
test = open("test.txt", "w+")
classes_file = open("classes.txt", "w+")


img_train = os.listdir(path_name + "Train/")
img_test = os.listdir(path_name + "Test/")

names = []

def get_class(img):
    global names
    name_img = img.split(" ")[0]

    for i, name in enumerate(names):
        if name_img == name:
            return i

def make_txt(name, img_array):
    for img in img_array:
        name.write(img +", "+ str(get_class(img)) +"\n")


def make_txt_classes(name):
    global names
    for nm in names:
        name.write(nm +"\n")

def make_classes(img_array):
    global names
    for img in img_array:
        names.append(img.split(" ")[0])

    names = set(names)


make_classes(img_train)

make_txt(train, img_train)
make_txt(test, img_test)
make_txt_classes(classes_file)