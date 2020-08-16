import imutils
import cv2
import numpy as np


def resize_to_fit(image, box_size):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """
    (h, w) = image.shape[:2]
    if w > h:
        image = imutils.resize(image, width=box_size)
    else:
        image = imutils.resize(image, height=box_size)
    padW = int((box_size - image.shape[1]) / 2.0)
    padH = int((box_size - image.shape[0]) / 2.0)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (box_size, box_size))
    return image


def image_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    return cv2.threshold(gray, 0, 250, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def letter_preprocessing(image, box_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the letter so it fits in a box
    image = resize_to_fit(image, box_size)
    # Add a third channel dimension to the image to make Keras happy
    return np.expand_dims(image, axis=2)
