import requests
import base64
import uuid
import os
import argparse

from tqdm import tqdm
from stem import Signal
from stem.control import Controller
from json import JSONDecodeError


class Tor:
    def __init__(self, session):
        self.session = session

    @classmethod
    def get_session(cls):
        with Controller.from_port(port=9051) as controller:
            controller.authenticate(password='password')
            controller.signal(Signal.NEWNYM)
            session = requests.session()
            session.proxies = {'http': 'socks5://127.0.0.1:9050',
                               'https': 'socks5://127.0.0.1:9050'}
            return cls(session)

    def get(self, url, **kwargs):
        return self.session.get(url, **kwargs)

    def check_ip(self):
        return self.get('https://api.ipify.org').text


def refresh_session():
    return Tor.get_session()


def download_examples(folder, num_samples):
    num_saved = 0
    pbar = tqdm(total = num_samples)
    while num_saved < num_samples:
        tsession = refresh_session()
        try:
            img_bytes = tsession.get('https://is.fssp.gov.ru/refresh_visual_captcha/').json()['image']
            num_saved += 1
        except JSONDecodeError:
            tsession = refresh_session()
            continue
        with open(os.path.join(folder, str(uuid.uuid4()) + '.jpg'), 'wb') as f:
            f.write(base64.b64decode(img_bytes.split(',')[1]))
        pbar.update(1)
    pbar.close()
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help='Saving folder', type=str)
    parser.add_argument("-n", "--num-examples", help='Number of examples', type=int, default=100, nargs='?', required=False)
    args = parser.parse_args()
    download_examples(args.folder, args.num_examples)
    