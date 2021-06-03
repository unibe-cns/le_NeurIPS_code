import numpy as np
from itertools import permutations


class BarsDataset(object):
    def __init__(self, square_size, bottom_left=0.0, top_right=1.0, noise_level=1e-2, samples_per_class=10, seed=42):
        if seed is not None:
            np.random.seed(seed)
        debug = False
        self.__vals = []
        self.__cs = []
        self.class_names = ['horiz', 'vert', 'diag']
        ones = list(np.ones(square_size) + (top_right - 1.))
        if debug:
            print(ones)
        starter = [ones]
        for i in range(square_size - 1):
            starter.append(list(np.zeros(square_size) + bottom_left))
        if debug:
            print('Starter')
            print(starter)
        horizontals = []
        for h in permutations(starter):
            horizontals.append(list(h))
        horizontals = np.unique(np.array(horizontals), axis=0)
        if debug:
            print('Horizontals')
            print(horizontals)
        verticals = []
        for h in horizontals:
            v = np.transpose(h)
            verticals.append(v)
        verticals = np.array(verticals)
        if debug:
            print('Verticals')
            print(verticals)
        diag = [top_right - bottom_left for i in range(square_size)]
        first = np.diag(diag) + bottom_left
        second = first[::-1]
        diagonals = [first, second]
        if debug:
            print('Diagonals')
            print(diagonals)
        n = 0
        idx = 0
        while n < samples_per_class:
            h = horizontals[idx].flatten()
            h = list(h + np.random.rand(len(h))*noise_level)
            self.__vals.append(h)
            self.__cs.append(0)
            n += 1
            idx += 1
            if idx >= len(horizontals):
                idx = 0
        n = 0
        idx = 0
        while n < samples_per_class:
            v = verticals[idx].flatten()
            v = list(v + np.random.rand(len(v))*noise_level)
            self.__vals.append(v)
            self.__cs.append(1)
            n += 1
            idx += 1
            if idx >= len(verticals):
                idx = 0
        n = 0
        idx = 0
        while n < samples_per_class:
            d = diagonals[idx].flatten()
            d = list(d + np.random.rand(len(d))*noise_level)
            self.__vals.append(d)
            self.__cs.append(2)
            n += 1
            idx += 1
            if idx >= len(diagonals):
                idx = 0

    def __getitem__(self, index):
        return np.array(self.__vals[index]), np.array(self.__cs[index])

    def __len__(self):
        return len(self.__cs)

