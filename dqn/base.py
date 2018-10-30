import inspect
import os
import pprint

import tensorflow as tf
from tqdm import tqdm
from dqn.utils import bcolors

pp = pprint.PrettyPrinter().pprint


def class_vars(obj):
    return {k: v for k, v in inspect.getmembers(obj)
            if not k.startswith('__') and not callable(k)}


class BaseModel(object):
    """Abstract object representing an Reader model."""

    def __init__(self, config):
        self._saver = None
        self.config = config

        try:
            self._attrs = config.__dict__['__flags']
        except:
            self._attrs = class_vars(config)
        pp(self._attrs)

        self.config = config

        for attr in self._attrs:
            name = attr if not attr.startswith('_') else attr[1:]
            setattr(self, name, getattr(self.config, attr))

    def save_model(self, step=None):
        tqdm.write("[*] Saving checkpoints...")
        model_name = type(self).__name__

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

    def load_model(self):
        print(bcolors.OKBLUE + "[*] Loading checkpoints..." + bcolors.ENDC)

        ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
        if ckpt:
            self.saver.restore(self.sess, ckpt)
            print(bcolors.OKGREEN + "[+] Load Success: {}".format(ckpt) + bcolors.ENDC)
            return True
        else:
            print(bcolors.FAIL + "[-] Load FAILED: {}".format(self.checkpoint_dir + bcolors.ENDC))
            return False

    @property
    def checkpoint_dir(self):
        return os.path.join('checkpoints', self.model_dir)

    @property
    def model_dir(self):
        model_dir = self.config.env_name
        for k, v in self._attrs.items():
            if not k.startswith('_') and k not in ['display']:
                model_dir += "/%s-%s" % (k, ",".join([str(i) for i in v]) if type(v) == list else v)
        return model_dir + '/'

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver
