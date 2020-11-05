import os
import cv2
import random
import imutils
import time
import shutil
import numpy as np
from utils.utils import resize_to_fit, image_preprocessing


def get_separate_letters(image):
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1] if imutils.is_cv3() else contours[0]

    letter_image_regions = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    letters = []
    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box
        letter_image = image[y - 2:y + h + 2, x - 2:x + w + 2]
        if letter_image.shape[0] > 10 and letter_image.shape[1] > 10:
            letters.append(letter_image)
    return letters


def create_train_set(output_folder, markup_files):
    os.mkdir(output_folder)
    counts = {}
    for file in markup_files:
        labels = [x for x in file.split('.png')[0].split('/')[-1]]
        image = cv2.imread(file)
        thresh = image_preprocessing(image)
        letters = get_separate_letters(thresh)
        for letter, label in zip(letters, labels):
            save_path = os.path.join(output_folder, label) 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            count = counts.get(label, 1)
            p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
            cv2.imwrite(p, letter)
            counts[label] = count + 1
