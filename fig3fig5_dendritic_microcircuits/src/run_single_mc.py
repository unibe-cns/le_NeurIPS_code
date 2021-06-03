import sys
import datetime
import time
import dataset
import numpy as np
import experiment as e
import network as pyral
import matplotlib.pyplot as plt


def plot_selfpred_ws(experiment, dirname, filename, save):
    save_name = filename + '_selfpred_layer_0_W_ip.npy'
    w_ip = experiment.load_result(save_name)
    save_name = filename + '_selfpred_layer_0_W_pi.npy'
    w_pi = experiment.load_result(save_name)
    save_name = filename + '_selfpred_layer_1_W_up.npy'
    w_up = experiment.load_result(save_name)
    save_name = filename + '_selfpred_layer_0_W_down.npy'
    w_down = experiment.load_result(save_name)
    save_name = filename + '_selfpred_time.npy'
    t = experiment.load_result(save_name)
    fig = plt.figure()
    plt.title("Weight evolution during selfprediting")
    plt.plot(t, w_up[:, 0, 0], label="W_up", alpha=0.7, color='C0')
    plt.plot(t, w_ip[:, 0, 0], label="W_ip", alpha=0.9, color='C0', ls='--')
    plt.plot(t, w_down[:, 0, 0], label="W_down", alpha=0.7, color='C1')
    plt.plot(t, -1 * w_pi[:, 0, 0], label="-1 * w_pi", alpha=0.9, color='C1', ls='--')
    plt.xlabel("t [ms]")
    plt.ylabel("weight")
    plt.legend()
    save_name = '../experiment_results/' + dirname + '/' + filename + '_selfpred_ws.png'
    plt.savefig(save_name)
    if save:
        save_name = '../experiment_results/for_plot_selfpred_w_ip.npy'
        np.save(save_name, np.array([t, w_ip[:, 0, 0]]))
        save_name = '../experiment_results/for_plot_selfpred_w_up.npy'
        np.save(save_name, np.array([t, w_up[:, 0, 0]]))
        save_name = '../experiment_results/for_plot_selfpred_w_pi.npy'
        np.save(save_name, np.array([t, w_pi[:, 0, 0]]))
        save_name = '../experiment_results/for_plot_selfpred_w_down.npy'
        np.save(save_name, np.array([t, w_down[:, 0, 0]]))
    return


def plot_selfpred_volts_comp(experiment, dirname, filename, save):
    save_name = filename + '_selfpred_layer_1_pyr_forw.npy'
    pyr_forw = experiment.load_result(save_name)
    save_name = filename + '_selfpred_layer_0_inn_forw.npy'
    inn_forw = experiment.load_result(save_name)
    save_name = filename + '_selfpred_time.npy'
    t = experiment.load_result(save_name)
    fig = plt.figure()
    plt.title("Selfpredicion pred. volts comparison")
    plt.plot(t, pyr_forw[:, 0], label="u_pyr_top_forw", alpha=0.7, color='C0')
    plt.plot(t, inn_forw[:, 0], label="u_inn_bottom_forw", alpha=0.7, color='C1')
    plt.xlabel("t [ms]")
    plt.ylabel("u_pred")
    plt.legend()
    save_name = '../experiment_results/' + dirname + '/' + filename + '_selfpred_volts_comp.png'
    plt.savefig(save_name)
    if save:
        save_name = '../experiment_results/for_plot_selfpred_pyr_forw.npy'
        np.save(save_name, np.array([t, pyr_forw[:, 0]]))
        save_name = '../experiment_results/for_plot_selfpred_inn_forw.npy'
        np.save(save_name, np.array([t, inn_forw[:, 0]]))
    return


def plot_selfpred_apical(experiment, dirname, filename, save):
    save_name = filename + '_selfpred_layer_0_pyr_apical.npy'
    v_apical = experiment.load_result(save_name)
    save_name = filename + '_selfpred_time.npy'
    t = experiment.load_result(save_name)
    fig = plt.figure()
    plt.title("Selfpredicion apical voltage")
    plt.plot(t, v_apical[:, 0], label="v_apical", alpha=1.0, color='C0')
    plt.xlabel("t [ms]")
    plt.ylabel("v_apical")
    plt.legend()
    save_name = '../experiment_results/' + dirname + '/' + filename + '_selfpred_apical.png'
    plt.savefig(save_name)
    if save:
        save_name = '../experiment_results/for_plot_selfpred_pyr_apical.npy'
        np.save(save_name, np.array([t, v_apical[:, 0]]))
    return


def plot_loss(experiment, dirname, filename, save):
    save_name = filename + '_train_validation_results.npy'
    val_res = experiment.load_result(save_name)
    # validations can happen more than once per epoch
    # if epochs should be on x-axis, need to translate from
    # seen training patterns to number of training epochs
    n_epochs = experiment.training_params['n_epochs']
    n_validations = len(val_res)
    # how much of an epoch has passed after one eval
    factor = float(n_epochs) / n_validations
    x_vals = np.array(range(len(val_res[:, 0]))) * factor
    fig = plt.figure()
    plt.title("Validation error during training")
    plt.semilogy(x_vals, val_res[:, 1], label="loss")
    plt.xlabel("epoch")
    plt.ylabel("mse-loss")
    save_name = '../experiment_results/' + dirname + '/' + filename + '_train_val_acc.png'
    plt.savefig(save_name)
    if save:
        save_name = '../experiment_results/for_plot_losses.npy'
        np.save(save_name, np.array([x_vals, val_res[:, 1]]))


def plot_test_eval(experiment, dirname, filename, pretrain, save):
    mode = 'test'
    if pretrain:
        save_name = filename + '_pretrain_eval_{0}_layer_{1}_pyr_forw.npy'.format(mode, len(experiment.network_params['dims'])-2)
    else:
        save_name = filename + '_eval_{0}_layer_{1}_pyr_forw.npy'.format(mode, len(experiment.network_params['dims'])-2)
    u_out = experiment.load_result(save_name)
    if pretrain:
        save_name = filename + '_pretrain_eval_{0}_u_tgt.npy'.format(mode)
    else:
        save_name = filename + '_eval_{0}_u_tgt.npy'.format(mode)
    u_tgt = experiment.load_result(save_name)
    fig = plt.figure()
    if pretrain:
        title = 'Output vs Target: pre training'
    else:
        title = 'Output vs Target: post training'
    plt.title(title)
    plt.plot(u_out[:, 0], label='u_out')
    plt.plot(u_tgt[:, 0], label='u_tgt')
    plt.legend()
    if pretrain:
        save_name = '../experiment_results/' + dirname + '/' + filename + '_pretrain_eval_{}_u_out.png'.format(mode)
    else:
        save_name = '../experiment_results/' + dirname + '/' + filename + '_eval_{}_u_out.png'.format(mode)
    plt.savefig(save_name)
    if save:
        if pretrain:
            save_name = '../experiment_results/for_plot_pretrain_eval_u_out.npy'
        else:
            save_name = '../experiment_results/for_plot_final_eval_u_out.npy'
        np.save(save_name, u_out[:, 0])
        if pretrain:
            save_name = '../experiment_results/for_plot_pretrain_eval_u_tgt.npy'
        else:
            save_name = '../experiment_results/for_plot_final_eval_u_tgt.npy'
        np.save(save_name, u_tgt[:, 0])


def plot_training_ws(experiment, dirname, filename, save):
    save_name = filename + '_train_layer_0_W_ip.npy'
    w_ip = experiment.load_result(save_name)
    save_name = filename + '_train_layer_0_W_up.npy'
    w_up_0 = experiment.load_result(save_name)
    save_name = filename + '_train_layer_1_W_up.npy'
    w_up_1 = experiment.load_result(save_name)
    save_name = filename + '_train_time.npy'
    t = experiment.load_result(save_name)
    save_name = filename + '_w_tgt.npy'
    w_tgt = experiment.load_result(save_name)
    fig = plt.figure()
    plt.title("Weight evolution during training")
    plt.plot(t, w_up_1[:, 0, 0], label="W_up_1", alpha=0.7, color='C0')
    plt.plot(t, w_ip[:, 0, 0], label="W_ip", alpha=0.7, color='C0', ls='--')
    plt.axhline(w_tgt[1], label="W_tgt_1", alpha=0.9, color='C0', ls=':')
    plt.plot(t, w_up_0[:, 0, 0], label="W_up_0", alpha=0.7, color='C1')
    plt.axhline(w_tgt[0], label="W_tgt_0", alpha=0.9, color='C1', ls=':')
    plt.xlabel("t [ms]")
    plt.ylabel("weight")
    plt.legend()
    save_name = '../experiment_results/' + dirname + '/' + filename + '_train_ws.png'
    plt.savefig(save_name)
    if save:
        save_name = '../experiment_results/for_plot_train_w_ip.npy'
        np.save(save_name, np.array([t, w_ip[:, 0, 0]]))
        save_name = '../experiment_results/for_plot_train_w_up_0.npy'
        np.save(save_name, np.array([t, w_up_0[:, 0, 0]]))
        save_name = '../experiment_results/for_plot_train_w_up_1.npy'
        np.save(save_name, np.array([t, w_up_1[:, 0, 0]]))
        save_name = '../experiment_results/for_plot_train_w_tgt.npy'
        np.save(save_name, np.array(w_tgt))
    return


def plot_training_apical(experiment, dirname, filename, save):
    save_name = filename + '_train_layer_0_pyr_apical.npy'
    v_apical = experiment.load_result(save_name)
    save_name = filename + '_train_time.npy'
    t = experiment.load_result(save_name)
    fig = plt.figure()
    plt.title("Training apical voltage")
    plt.plot(t, v_apical[:, 0], label="v_apical", alpha=1.0, color='C0')
    plt.xlabel("t [ms]")
    plt.ylabel("v_apical")
    plt.legend()
    save_name = '../experiment_results/' + dirname + '/' + filename + '_train_apical.png'
    plt.savefig(save_name)
    if save:
        save_name = '../experiment_results/for_plot_train_pyr_apical.npy'
        np.save(save_name, np.array([t, v_apical[:, 0]]))
    return


def plot_training_pyr_volts(experiment, dirname, filename, save):
    save_name = filename + '_train_layer_0_pyr_basal.npy'
    v_basal = experiment.load_result(save_name)
    save_name = filename + '_train_layer_0_pyr_forw.npy'
    u_pyr = experiment.load_result(save_name)
    save_name = filename + '_train_time.npy'
    t = experiment.load_result(save_name)
    f = experiment.network_params['gb'] / (
            experiment.network_params['gb'] + experiment.network_params['gl'] +
            experiment.network_params['ga'])
    fig = plt.figure()
    plt.title("Training pyr voltages")
    plt.plot(t, v_basal[:, 0] * f, label="v_basal*", alpha=0.8, color='C0')
    plt.plot(t, u_pyr[:, 0], label="u_pyr", alpha=0.8, color='C1')
    plt.xlabel("t [ms]")
    plt.ylabel("voltage")
    plt.legend()
    save_name = '../experiment_results/' + dirname + '/' + filename + '_train_pyr_volts.png'
    plt.savefig(save_name)


def plot_training_pyr_error(experiment, dirname, filename, save):
    save_name = filename + '_train_layer_0_pyr_basal.npy'
    v_basal = experiment.load_result(save_name)
    save_name = filename + '_train_layer_0_pyr_forw.npy'
    u_pyr = experiment.load_result(save_name)
    save_name = filename + '_train_time.npy'
    t = experiment.load_result(save_name)
    f = experiment.network_params['gb'] / (
            experiment.network_params['gb'] + experiment.network_params['gl'] +
            experiment.network_params['ga'])
    fig = plt.figure()
    plt.title("Training pyr voltage diffs")
    plt.plot(t, u_pyr[:, 0] - v_basal[:, 0] * f, label="u_pyr - v_basal*", alpha=0.8, color='C0')
    plt.axhline(0., alpha=0.4)
    plt.xlabel("t [ms]")
    plt.ylabel("voltage diff")
    plt.legend()
    save_name = '../experiment_results/' + dirname + '/' + filename + '_train_pyr_error.png'
    plt.savefig(save_name)


def plot_all(experiment, dirname, filename, show=False, save_for_plt=False):
    plot_selfpred_ws(experiment, dirname, filename, save_for_plt)
    plot_selfpred_volts_comp(experiment, dirname, filename, save_for_plt)
    plot_selfpred_apical(experiment, dirname, filename, save_for_plt)
    plot_loss(experiment, dirname, filename, save_for_plt)
    plot_test_eval(experiment, dirname, filename, True, save_for_plt)
    plot_test_eval(experiment, dirname, filename, False, save_for_plt)
    plot_training_ws(experiment, dirname, filename, save_for_plt)
    plot_training_apical(experiment, dirname, filename, save_for_plt)
    plot_training_pyr_volts(experiment, dirname, filename, save_for_plt)
    plot_training_pyr_error(experiment, dirname, filename, save_for_plt)
    if show:
        plt.show()


def save_pretrain_results(experiment, results_trainset, results_testset, eval_train, eval_test):
    records, T, r_in, u_tgt, out_seq = results_trainset
    test_req_quants = [[], ['pyr_forw']]
    for layer in range(len(test_req_quants)):
        for var in test_req_quants[layer]:
            save_name = experiment.filename + '_pretrain_eval_train_layer_{0}_{1}.npy'.format(layer, var)
            vals = records[layer][var].data
            experiment.save_result(save_name, vals)
    save_name = experiment.filename + '_pretrain_eval_train_time.npy'
    experiment.save_result(save_name, T)
    save_name = experiment.filename + '_pretrain_eval_train_r_in.npy'
    experiment.save_result(save_name, r_in)
    save_name = experiment.filename + '_pretrain_eval_train_u_tgt.npy'
    experiment.save_result(save_name, u_tgt)
    save_name = experiment.filename + '_pretrain_eval_train_loss_acc.npy'
    experiment.save_result(save_name, eval_train)
    records, T, r_in, u_tgt, out_seq = results_testset
    for layer in range(len(test_req_quants)):
        for var in test_req_quants[layer]:
            save_name = experiment.filename + '_pretrain_eval_test_layer_{0}_{1}.npy'.format(layer, var)
            vals = records[layer][var].data
            experiment.save_result(save_name, vals)
    save_name = experiment.filename + '_pretrain_eval_test_time.npy'
    experiment.save_result(save_name, T)
    save_name = experiment.filename + '_pretrain_eval_test_r_in.npy'
    experiment.save_result(save_name, r_in)
    save_name = experiment.filename + '_pretrain_eval_test_u_tgt.npy'
    experiment.save_result(save_name, u_tgt)
    save_name = experiment.filename + '_pretrain_eval_test_loss_acc.npy'
    experiment.save_result(save_name, eval_test)
    return


def generate_targets(config_path, tgt_seed, n_samples=6):
    network_params, training_params = e.load_config(config_path, None, loading_template=True)
    print('Generating teacher network')
    tgtNet = pyral.Net(network_params, seed=tgt_seed)
    tgtNet.reflect()
    print('W_tgt_up_bottom', tgtNet.layer[0].W_up)
    print('W_tgt_up_top', tgtNet.layer[1].W_up)

    train_in = np.random.rand(n_samples, 1)
    test_in = train_in[:]
    train_tgt = np.zeros((n_samples, 1))
    test_tgt = np.zeros((n_samples, 1))

    rec_quants = [["pyr_forw", "pyr_apical", "inn_forw", "W_up", "W_ip"], ["pyr_forw", "W_up"]]
    result = tgtNet.evaluate(train_in, train_tgt, test_in, test_tgt, 1, classify=False,
                             rec_dt=network_params["dt"], rec_quants=rec_quants)
    ret_train, _, ret_test, _ = result
    recordings_train = ret_train[0]
    recordings_test = ret_test[0]
    # check that weights have not changed
    assert recordings_train[0]["W_up"].data[0][0][0] == recordings_train[0]["W_up"].data[-1][0][0]
    assert recordings_train[1]["W_up"].data[0][0][0] == recordings_train[1]["W_up"].data[-1][0][0]
    assert recordings_train[0]["W_up"].data[0][0][0] == recordings_test[0]["W_up"].data[-1][0][0]
    assert recordings_train[1]["W_up"].data[0][0][0] == recordings_test[1]["W_up"].data[-1][0][0]
    pres_steps = int(network_params["t_pattern"] / network_params["dt"])
    r_in = ret_test[2][::pres_steps]
    u_out_teach = ret_test[4]
    return r_in, u_out_teach, [tgtNet.layer[0].W_up, tgtNet.layer[1].W_up]


if __name__ == '__main__':
    mode = sys.argv[1]

    filename = 'mimic'

    if mode == 'train':
        config_path = sys.argv[2]
        config_name = config_path.split('/')[-1]
        config_name = config_name.split('.')[0]
        dirname = '{0}_{1:%Y-%m-%d_%H-%M-%S}'.format(config_name, datetime.datetime.now())
        network_params, training_params = e.load_config(config_path, None, loading_template=True)

        # the setting up of the datastream into the network uses a different seed than the one in the
        # config file to ensure, that all differently seeded networks are trained on the same input stream
        tgt_seed = 42
        r_in, u_tgt, w_tgts = generate_targets(config_path, tgt_seed)

        experiment = e.Experiment(dirname, filename, config_path)
        experiment.full_setup()
        save_name = filename + '_w_tgt.npy'
        experiment.save_result(save_name, w_tgts)
        print('Weights pre selfprediction training')
        print('W_stud_up_bottom', experiment.net.layer[0].W_up)
        print('W_stud_up_top', experiment.net.layer[1].W_up)
        print('W_stud_ip', experiment.net.layer[0].W_ip)
        print('W_stud_backw', experiment.net.layer[0].W_down)
        print('W_stud_pi', experiment.net.layer[0].W_pi)
        experiment.selfpredict(r_in, u_tgt)
        print('Weights post selfprediction training')
        print('W_stud_up_bottom', experiment.net.layer[0].W_up)
        print('W_stud_up_top', experiment.net.layer[1].W_up)
        print('W_stud_ip', experiment.net.layer[0].W_ip)
        print('W_stud_backw', experiment.net.layer[0].W_down)
        print('W_stud_pi', experiment.net.layer[0].W_pi)
        # make pre_training eval
        pre_train_eval = experiment.test(r_in, u_tgt, r_in, u_tgt, save=False)
        res_train, eval_train, res_test, eval_test = pre_train_eval
        save_pretrain_results(experiment, res_train, res_test, eval_train, eval_test)
        # train on task
        t_start = time.time()
        experiment.train(r_in, u_tgt, r_in, u_tgt)
        t_end = time.time()
        duration = t_end - t_start
        print('Training duration: {0} seconds'.format(duration))
        # make post_training eval
        experiment.test(r_in, u_tgt, r_in, u_tgt)
    if mode == 'eval':
        dirname = sys.argv[2]
        experiment = e.Experiment(dirname, filename, None)
        plot_all(experiment, dirname, filename, show=True, save_for_plt=False)
