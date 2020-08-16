import keras
import pickle
import os.path
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from utils.utils import image_preprocessing, letter_preprocessing
from utils.markup import get_separate_letters


class NN:
    box_size = 20
    MODEL_FILENAME = "captcha_model.pkl"
    MODEL_LABELS_FILENAME = "model_labels.pkl"
    
    def __init__(self, model, binarizer):
        self.model = model
        self.binarizer = binarizer
        
    def solve(self, image):
        image = image_preprocessing(image)
        letters = get_separate_letters(image)
        predictions = []
        for letter_image in letters:
            letter_image = cv2.cvtColor(letter_image, cv2.COLOR_GRAY2BGR)
            letter_image = letter_preprocessing(letter_image, self.box_size)
            letter_image = np.expand_dims(letter_image, axis=0)
            prediction = self.predict(letter_image)
            letter = self.binarizer.inverse_transform(prediction)[0]
            predictions.append(letter)
        return "".join(predictions)

    
    @classmethod
    def build(cls, alphabet_size, metrics=['accuracy'], optimizer="adam", loss="categorical_crossentropy"):
        model = Sequential()
        # First convolutional layer with max pooling
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=(cls.box_size, cls.box_size, 1), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Second convolutional layer with max pooling
        model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Hidden layer with 500 nodes
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))
        # Output layer with 32 nodes (one for each possible letter/number we predict)
        model.add(Dense(alphabet_size, activation="softmax"))
        # Ask Keras to build the TensorFlow model behind the scenes
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return cls(model, None)
    
    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'labels.pkl'), "rb") as f:
            binarizer = pickle.load(f)
        model = keras.models.load_model(os.path.join(path, 'model.pkl'))
        return cls(model, binarizer)
    
    def _binarize_labels(self, Y_train, Y_test):
        lb = LabelBinarizer().fit(Y_train)
        self.binarizer = lb 
        return lb.transform(Y_train), lb.transform(Y_test)
    
    def fit(self, X_train, X_test, Y_train, Y_test, batch_size=21, epochs=10, verbose=1):
        Y_train, Y_test = self._binarize_labels(Y_train, Y_test)
        self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), 
                       batch_size=batch_size, epochs=epochs, verbose=verbose)
        
    def predict(self, letter_image):
        return self.model.predict(letter_image)
        
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'labels.pkl'), "wb") as f:
            pickle.dump(self.binarizer, f)
        self.model.save(os.path.join(path, 'model.pkl'))
