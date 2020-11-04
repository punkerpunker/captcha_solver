import connexion
import base64
import numpy as np
import cv2
import os
from utils.model import NN


trained_model_location = 'trained_model'
model = NN.load(trained_model_location)


def solve(image):
    nparr = np.fromstring(base64.b64decode(image), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return model.solve(image)


app = connexion.App(__name__, specification_dir='./')
app.add_api('swagger.yaml')
app.run(port=os.environ.get('PORT'))
