import numpy as np


class Tracker:
    '''
    Tracks/records changes in 'target' array. Records 'length'*'compress_len' samples,
     compressed (averaged) into 'length' samples. The result is stored in 'data'.
     Note that the first value in the 'data' is already the average of multiple values of the target array.
     If 'compress_len' is not 1 the initial value of 'target' is therefore not equal to the first entry in 'data'.
     After recording call finalize to also add the remaining data in buffer to 'data' (finish the last compression).
     '''

    def __init__(self, length, target, compress_len):
        self.target = target
        self.data = np.zeros(tuple([length]) + target.shape, dtype=np.float32)
        self.index = 0
        self.buffer = np.zeros(target.shape)
        self.din = compress_len

    def record(self):
        self.buffer += self.target
        if (self.index + 1) % self.din == 0:
            self.data[int(self.index / self.din), :] = self.buffer / self.din
            self.buffer.fill(0)
        self.index += 1

    def finalize(self):
        '''fill last data point with average of remaining target data in buffer.'''
        n_buffer = self.index % self.din
        if n_buffer > 0:
            self.data[int(self.index / self.din), :] = self.buffer / n_buffer


def soft_relu(x, thresh=15):
    res = x.copy()
    ind = np.abs(x) < thresh
    res[x < -thresh] = 0
    res[ind] = np.log(1 + np.exp(x[ind]))
    return res

# faster than the stable version
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_stable(x, thresh=15):
    res = np.ones_like(x)
    ind = np.abs(x) < thresh
    res[x < -thresh] = 0
    res[ind] =  1 / (1 + np.exp(-x[ind]))
    return res

def time_str(sec):
    string = ""
    h = int(sec / 3600)
    if h > 0:
        string = str(h) + "h, "
    if int(sec / 60) > 0:
        m = int((sec - h * 3600) / 60)
        string += str(m) + "min and "
    string += str(int(sec % 60)) + "s"
    return string


def ewma(data, window):
    '''
    Exponentially weighted moving average
    '''
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = data.shape[0]
    pows = alpha_rev ** (np.arange(n + 1))
    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)
    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def accuracy(pred, true):
    pred_class = np.argmax(pred, axis=1)
    true_class = np.argmax(true, axis=1)
    acc = float(np.sum(pred_class == true_class)) / len(pred)
    return "accuracy", acc


