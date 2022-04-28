# library
import numpy as np
from matplotlib import pyplot as plt
from collections.abc import Iterable
import sounddevice as sd
import json
import torch
import torch

# local



################
## Audio 관련 ##
################

def listen_json(path,sr=22050):
    with open(path, 'r') as f:
            waveform = torch.FloatTensor(json.load(f)['audio'])
    try:
        sd.play(waveform, sr, blocking=True)
    except:
        sd.play(waveform[0],sr,blocking=True)




def listen(audio,fs=48000):
    '''
    "audio" is waveform which type is torch.tensor
    audio shape : [fs], sampling rate = 22050
    '''
    sd.play(audio, 16000, blocking=True)



###########
# logging #
###########
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0  # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        if val != None:  # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val ** 2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2 / self.count - self.avg ** 2)
        else:
            pass



class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try:
            return len(self.read())
        except:
            return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.', v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log

def draw_curve(work_dir, train_logger, test_logger):
        train_logger = train_logger.read()
        test_logger = test_logger.read()
        epoch, train_loss = zip(*train_logger)
        epoch,test_loss = zip(*test_logger)

        plt.plot(epoch, train_loss, color='blue', label="Train Loss")
        plt.plot(epoch, test_loss, color='red', label="Test Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(work_dir + '/loss_curve.png')
        plt.close()




if __name__ == '__main__':
    ## Test listen_json()
    listen_json('./data/fma_large_json/000002.json')