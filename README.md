# Captcha Solver
This module built to simplify the process of training NN to solve text captcha. 

## Using of pre-existing model for captcha from fssp.gov.ru

Installation of captcha-solver api with pre-trained model for captcha from fssp.gov.ru could be as simple as:

```
git clone git@github.com:punkerpunker/captcha_solver.git
cd captcha_solver
docker-compose up -d --build captcha-solver
```
This command registers api on 2103 port.

Model works with images in string base64 format. You can pass image to API as a query argument with example below:
```python
import requests
from matplotlib import pyplot as plt

url = 'http://0.0.0.0:2103/solve?'

img_bytes = requests.get('https://is.fssp.gov.ru/refresh_visual_captcha/').json()['image'].split(',')[1]

params = {'image': img_bytes}
resp = requests.get(url, params=params)

resp.json()

>>> 'л9м45'
```

Full example can be founded in test_app.ipynb:

![alt text](https://github.com/punkerpunker/captcha_solver/blob/master/example.jpg)
