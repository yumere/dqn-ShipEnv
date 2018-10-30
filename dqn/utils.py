import sys
import time

import numpy as np

if (sys.version_info[0] == 2):
    import cPickle
elif (sys.version_info[0] == 3):
    import _pickle as cPickle

try:
    from skimage.transform import resize
except:
    import cv2

    resize = cv2.resize


def rgb2gray(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])


def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result

    return timed


def get_time():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())


@timeit
def save_pkl(obj, path):
    with open(path, 'w') as f:
        cPickle.dump(obj, f)
        print("  [*] save %s" % path)


@timeit
def load_pkl(path):
    with open(path) as f:
        obj = cPickle.load(f)
        print("  [*] load %s" % path)
        return obj


@timeit
def save_npy(obj, path):
    np.save(path, obj)
    print("  [*] save %s" % path)


@timeit
def load_npy(path):
    obj = np.load(path)
    print("  [*] load %s" % path)
    return obj


class bcolors(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'