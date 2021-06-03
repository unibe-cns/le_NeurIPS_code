import sys
import datetime
import time
import dataset
import numpy as np
import experiment as e


if __name__ == '__main__':
    mode = sys.argv[1]

    filename = 'bars'

    if mode == 'train':
        config_path = sys.argv[2]
        config_name = config_path.split('/')[-1]
        config_name = config_name.split('.')[0]
        dirname = '{0}_{1:%Y-%m-%d_%H-%M-%S}'.format(config_name, datetime.datetime.now())

        x_train, y_train = dataset.BarsDataset(3, samples_per_class=3, noise_level=0, seed=50)[:-1]
        x_val, y_val = dataset.BarsDataset(3, samples_per_class=3, noise_level=0)[:-1]
        x_test, y_test = dataset.BarsDataset(3, samples_per_class=3, noise_level=0)[:-1]

        experiment = e.Experiment(dirname, filename, config_path)
        experiment.full_setup()

        t_start = time.time()
        experiment.train(x_train, y_train, x_val, y_val)
        t_end = time.time()
        duration = t_end - t_start
        print('Training duration: {0} seconds'.format(duration))
        experiment.test(x_train, y_train, x_test, y_test)
    if mode == 'eval':
        dirname = sys.argv[2]
        experiment = e.Experiment(dirname, filename, None)
        plot_list = ['val_acc',
                     'eval_train_u_out',
                     'eval_test_u_out',
                     ]
        experiment.evaluate(plot_list, show=True)
