import os
import yaml
import subprocess
import numpy as np
import network as pyrnet
import utils as u
import matplotlib.pyplot as plt


def load_config(dirname, filename, loading_template=False, verbose=False):
    if loading_template:
        path = dirname
    else:
        path = dirname + '/' + filename + '_config.yaml'
    with open(path) as f:
        data = yaml.load_all(f, Loader=yaml.Loader)
        all_configs = next(iter(data))
    network_params = all_configs[0]
    training_params = all_configs[1]
    if verbose:
        print('network_params: ', network_params)
        print('training_params: ', training_params)
    return network_params, training_params


def save_config(dirname, filename, network_params, training_params, epoch_dir=(False, -1)):
    dirname = '../experiment_results/' + dirname
    if not dirname[-1] == '/':
        dirname += '/'
    if epoch_dir[0]:
        dirname += 'epoch_{}/'.format(epoch_dir[1])
    try:
        os.makedirs(dirname)
        print("Directory ", dirname, " Created ")
    except FileExistsError:
        print("Directory ", dirname, " already exists")
    # save parameter configs
    with open(dirname + filename + '_config.yaml', 'w') as f:
        yaml.dump([network_params, training_params], f)
    with open(dirname + filename + '_gitsha.txt', 'w') as f:
        f.write(subprocess.check_output(["git", "rev-parse", "HEAD"]).decode())
    return





class Experiment(object):
    def __init__(self, dirname, filename, config_path=None):
        self.dirname = dirname
        self.filename = filename
        # config path only given for fresh setups, others have already saved config in experiment dir
        self.config_path = config_path
        if self.config_path is None:
            self.network_params, self.training_params = load_config(self.dirname, self.filename)
        else:
            self.network_params, self.training_params = load_config(self.config_path, None, loading_template=True)
            # after loading template, now save used config in experiment dir
            save_config(self.dirname, self.filename, self.network_params, self.training_params)
        return

    def full_setup(self):
        self.net = pyrnet.Net(self.network_params, seed=self.training_params['seed'])
        if self.training_params['init_selfpred']:
            print("enforcing selfpredicting state")
            self.net.reflect()

    def load_setup(self):
        raise NotImplementedError

    def train(self, x_train, y_train, x_val, y_val):
        if self.training_params['eval_metric'] == 'accuracy':
            metric = u.accuracy
        else:
            metric = None
        # save pre-training weights
        filename = self.filename + '_pre_training_weights.npy'
        save_name = '../experiment_results/' + self.dirname + '/' + filename
        self.net.dump_weights(save_name)
        results = self.net.train(x_train, y_train, x_val, y_val, n_epochs=self.training_params['n_epochs'],
                val_len=self.training_params['n_val_samples'], vals_per_epoch=self.training_params['n_vals_per_epoch'],
                n_out=self.network_params['dims'][-1], classify=self.training_params['classify'],
                u_high=self.training_params['u_tgt_high'], u_low=self.training_params['u_tgt_low'],
                rec_quants=self.training_params['recorded_quants'], rec_dt=self.training_params['record_dt'],
                metric=metric, info_update=100, breadcrumbs=self.training_params['snapshot_epochs'])
        self.save_train_results(results)
        # save post-training weights
        filename = self.filename + '_post_training_weights.npy'
        save_name = '../experiment_results/' + self.dirname + '/' + filename
        self.net.dump_weights(save_name)

    def selfpredict(self, x_train, y_train):
        if self.training_params['eval_metric'] == 'accuracy':
            metric = u.accuracy
        else:
            metric = None
        assert self.training_params['init_selfpred'] is False
        self.net.update_eta(self.network_params['eta_selfpred'])
        # save pre-training weights
        filename = self.filename + '_pre_selfpred_weights.npy'
        save_name = '../experiment_results/' + self.dirname + '/' + filename
        self.net.dump_weights(save_name)
        results = self.net.selfpredict(x_train, y_train, n_epochs=self.training_params['n_epochs_selfpred'],
                n_out=self.network_params['dims'][-1], classify=self.training_params['classify'],
                u_high=self.training_params['u_tgt_high'], u_low=self.training_params['u_tgt_low'],
                rec_quants=self.training_params['recorded_quants'], rec_dt=self.training_params['record_dt'],
                metric=metric)
        self.save_selfpred_results(results)
        # save post-training weights
        filename = self.filename + '_post_selfpred_weights.npy'
        save_name = '../experiment_results/' + self.dirname + '/' + filename
        self.net.dump_weights(save_name)
        self.net.update_eta(self.network_params['eta'])

    def test(self, x_train, y_train, x_test, y_test, save=True):
        if self.network_params['latent_eq']:
            test_req_quants = [[], ['pyr_forw']]
        else:
            test_req_quants = [[], ['pyr_soma']]
        if self.training_params['eval_metric'] == 'accuracy':
            metric = u.accuracy
        else:
            metric = None
        results = self.net.evaluate(x_train, y_train, x_test, y_test, n_out=self.network_params['dims'][-1],
                classify=self.training_params['classify'], u_high=self.training_params['u_tgt_high'],
                u_low=self.training_params['u_tgt_low'], rec_quants=test_req_quants,
                rec_dt=self.training_params['record_dt'], metric=metric)
        res_train, eval_train, res_test, eval_test = results
        if save:
            self.save_test_results(res_train, res_test, eval_train, eval_test)
        return res_train, eval_train, res_test, eval_test

    def save_train_results(self, results):
        records, T, r_in, u_tgt, out_seq, val_res = results
        recorded_vars = self.training_params['recorded_quants']
        save_traces = self.training_params.get('save_traces', True)
        if save_traces:
            for layer in range(len(recorded_vars)):
                for var in recorded_vars[layer]:
                    save_name = self.filename + '_train_layer_{0}_{1}.npy'.format(layer, var)
                    vals = records[layer][var].data
                    self.save_result(save_name, vals)
            save_name = self.filename + '_train_time.npy'
            self.save_result(save_name, T)
            save_name = self.filename + '_train_r_in.npy'
            self.save_result(save_name, r_in)
            save_name = self.filename + '_train_u_tgt.npy'
            self.save_result(save_name, u_tgt)
        save_name = self.filename + '_train_validation_results.npy'
        self.save_result(save_name, val_res)
        return

    def save_selfpred_results(self, results):
        records, T, r_in, u_tgt, out_seq = results
        recorded_vars = self.training_params['recorded_quants']
        save_traces = self.training_params.get('save_traces', True)
        if save_traces:
            for layer in range(len(recorded_vars)):
                for var in recorded_vars[layer]:
                    save_name = self.filename + '_selfpred_layer_{0}_{1}.npy'.format(layer, var)
                    vals = records[layer][var].data
                    self.save_result(save_name, vals)
            save_name = self.filename + '_selfpred_time.npy'
            self.save_result(save_name, T)
            save_name = self.filename + '_selfpred_r_in.npy'
            self.save_result(save_name, r_in)
        return

    def save_test_results(self, results_trainset, results_testset, eval_train, eval_test):
        records, T, r_in, u_tgt, out_seq = results_trainset
        #save_traces = self.training_params.get('save_traces', True)
        if self.network_params['latent_eq']:
            test_req_quants = [[], ['pyr_forw']]
        else:
            test_req_quants = [[], ['pyr_soma']]
        for layer in range(len(test_req_quants)):
            for var in test_req_quants[layer]:
                save_name = self.filename + '_eval_train_layer_{0}_{1}.npy'.format(layer, var)
                vals = records[layer][var].data
                self.save_result(save_name, vals)
        save_name = self.filename + '_eval_train_time.npy'
        self.save_result(save_name, T)
        save_name = self.filename + '_eval_train_r_in.npy'
        self.save_result(save_name, r_in)
        save_name = self.filename + '_eval_train_u_tgt.npy'
        self.save_result(save_name, u_tgt)
        save_name = self.filename + '_eval_train_loss_acc.npy'
        self.save_result(save_name, eval_train)
        records, T, r_in, u_tgt, out_seq = results_testset
        for layer in range(len(test_req_quants)):
            for var in test_req_quants[layer]:
                save_name = self.filename + '_eval_test_layer_{0}_{1}.npy'.format(layer, var)
                vals = records[layer][var].data
                self.save_result(save_name, vals)
        save_name = self.filename + '_eval_test_time.npy'
        self.save_result(save_name, T)
        save_name = self.filename + '_eval_test_r_in.npy'
        self.save_result(save_name, r_in)
        save_name = self.filename + '_eval_test_u_tgt.npy'
        self.save_result(save_name, u_tgt)
        save_name = self.filename + '_eval_test_loss_acc.npy'
        self.save_result(save_name, eval_test)
        return

    def save_result(self, filename, result):
        np.save('../experiment_results/' + self.dirname + '/' + filename, result)
        return

    def load_result(self, filename):
        result = np.load('../experiment_results/' + self.dirname + '/' + filename)
        return result

    def evaluate(self, plot_list, show=False):
        if 'val_acc' in plot_list:
            self.plot_val_acc()
        if 'eval_train_u_out' in plot_list:
            self.plot_eval_u_out(True)
        if 'eval_test_u_out' in plot_list:
            self.plot_eval_u_out(False)
        if show:
            plt.show()

    def plot_val_acc(self):
        save_name = self.filename + '_train_validation_results.npy'
        val_res = self.load_result(save_name)
        # validations can happen more than once per epoch
        # if epochs should be on x-axis, need to translate from
        # seen training patterns to number of training epochs
        n_epochs = self.training_params['n_epochs']
        n_validations = len(val_res)
        # how much of an epoch has passed after one eval
        factor = float(n_epochs) / n_validations
        x_vals = np.array(range(len(val_res[:, 0]))) * factor
        fig = plt.figure()
        plt.title("Validation error during training")
        plt.semilogy(x_vals, val_res[:, 1], label="loss")
        plt.xlabel("epoch")
        plt.ylabel("mse-loss")
        ax2 = plt.gca().twinx()
        #ax2.plot(val_res[:, 0], pyral.ewma(val_res[:, 2], round(len(val_res) / 10)), c="g", label="accuracy")
        ax2.plot(x_vals, val_res[:, 2], c="g", label="accuracy")
        ax2.set_ylabel("accuracy")
        save_name = '../experiment_results/' + self.dirname + '/' + self.filename + '_train_val_acc.png'
        plt.savefig(save_name)

    def plot_eval_u_out(self, train):
        le = self.network_params['latent_eq']
        mode = 'test'
        if train:
            mode = 'train'
        if le:
            save_name = self.filename + '_eval_{0}_layer_{1}_pyr_forw.npy'.format(mode, len(self.network_params['dims'])-2)
            u_out = self.load_result(save_name)
        else:
            save_name = self.filename + '_eval_{0}_layer_{1}_pyr_soma.npy'.format(mode, len(self.network_params['dims'])-2)
            u_out = self.load_result(save_name)
        save_name = self.filename + '_eval_{0}_u_tgt.npy'.format(mode)
        u_tgt = self.load_result(save_name)
        n_out = len(u_tgt[0])
        fig, axes = plt.subplots(nrows=n_out, ncols=1)
        for i in range(n_out):
            axes[i].plot(u_out[:, i], label='u_out {}'.format(i))
            axes[i].plot(u_tgt[:, i], label='u_tgt {}'.format(i))
            axes[i].legend()
        save_name = '../experiment_results/' + self.dirname + '/' + self.filename + '_eval_{}_u_out.png'.format(mode)
        plt.savefig(save_name)



