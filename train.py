import cv2
import os
import numpy as np
import argparse
import shutil
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.model_selection import train_test_split
from utils.utils import letter_preprocessing
from imutils import paths
from utils.model import NN
from utils.markup import create_train_set


LETTER_IMAGES_FOLDER = "letters"


def train_model(letters_path, save_path):
    data = []
    labels = []

    for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
        image = cv2.imread(image_file)
        image = letter_preprocessing(image, NN.box_size)
        label = image_file.split(os.path.sep)[-2]
        data.append(image)
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1] (this improves training)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    alphabet_size = len(set(labels))

    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

    nn = NN.build(alphabet_size)
    nn.fit(X_train, X_test, Y_train, Y_test)
    return nn
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_set_location", help='Location with stored labelled captcha', type=str)
    parser.add_argument("-s", "--save-path", help='Path to save trained model', type=str, 
                        default='trained_model', nargs='?', required=False)
    args = parser.parse_args()
    markup_files = [os.path.join(args.train_set_location, x) for x in os.listdir(args.train_set_location)]
    create_train_set(LETTER_IMAGES_FOLDER, markup_files)
    nn = train_model(LETTER_IMAGES_FOLDER, args.save_path)
    shutil.rmtree(args.save_path)
    nn.save(args.save_path)
    print("Success")
