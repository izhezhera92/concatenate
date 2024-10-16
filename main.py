__author__ = "Ivan Zhezhera"
__date__ = "14.10.2024"


import cv2 
import time
import warnings
import logging
import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import random

import json
from json import JSONEncoder

import concate


ap = argparse.ArgumentParser()
ap.add_argument("-img", "--input_image", default="./img1/", required = False,
    help = "Path to the directory that contains the imags")
ap.add_argument("-res", "--resized", default="./resized/", required = False,
    help = "Path to the directory that contains the resized imags")
ap.add_argument("-p", "--persent", default=0.2, required = False,
    help = "Part of images to processing ")
args = vars(ap.parse_args())



warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S') 


image_path = args["input_image"]
resized_path = args["resized"]



def list_full_paths(path):
    return [os.path.join(path, file) for file in os.listdir(path)]


if __name__ == "__main__":
    cnct = concate.Concate(scale_factor = 0.5)

    #image_list = os.listdir(image_path)
    #image_list = list_full_paths(path = image_path)

    #logging.info(f"Resizing...")
    #for i in tqdm(range(len(image_list))):
    #    #cnct.resize_tiff_image_with_alpha(input_path = image_path + image_list[i], output_path = resized_path + image_list[i])  
    #    cnct.resize_tiff_image_with_alpha(input_path = image_list[i], output_path = image_list[i].replace(image_path, resized_path))

    tiff_paths = list_full_paths(path = resized_path)
    tiff_paths.reverse()
    print(tiff_paths)

    logging.info(f"Merging...")
    cnct.merge_large_tiffs_by_chunks(tiff_paths, chunk_size=(7000, 5000))