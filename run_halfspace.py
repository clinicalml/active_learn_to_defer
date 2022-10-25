import sys  
sys.path.insert(0, '')
import AL
import numpy as np
from matplotlib import pyplot as plt
import argparse
from argparse import Namespace

def log_args(args):
    """ print all args """
    lines = [' {:<25}- {}\n'.format(key, val) for key, val in vars(args).items()]
    for line in lines:
        print(line.rstrip())
        print('-------------------------------------------')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=10000, help='Number of iterations of train/test procedure.')
    parser.add_argument('--range-labels', type=int, default=20, help='Maximum number of samples that human could label.')
    ar = parser.parse_args()
    return ar


def preprocess_args(ar):
    if ar.log_dir is None:
        assert ar.log_name is not None

def run():
    configs     =   get_args()    
    main(**vars(configs))

def main(**kwargs):
    preprocess_args(Namespace(**kwargs))
    log_args(Namespace(**kwargs))    




    iter_size = Namespace(**kwargs).iters
    iter_size_inside_AL = 1
    range_labels = Namespace(**kwargs).range_labels

    for i in range(iter_size):
        kwargs = {"log_dir": str(i), "log_name": "data", "iters": iter_size_inside_AL, "range_labels": range_labels};
        AL.main(**kwargs)



    err_DoD = np.zeros([range_labels,])
    err_CAL = np.zeros([range_labels, ])
    err_staged = np.zeros([range_labels, ])
    err_joint = np.zeros([range_labels, ])
    err_DoD_std = np.zeros([range_labels,])
    err_CAL_std = np.zeros([range_labels, ])
    err_staged_std = np.zeros([range_labels, ])
    err_joint_std = np.zeros([range_labels, ])
    ITER = iter_size
    for i in range(ITER):
        datacal = np.load(str(i)+'/data/DataCAL.npy.npz')
        err_CAL += np.array(datacal['test_err_CAL'])
        err_CAL_std += np.array(datacal['test_err_CAL'])**2
        err_staged += np.array(datacal['test_err_staged_ERM'])
        err_staged_std += np.array(datacal['test_err_staged_ERM'])**2
        data = np.load(str(i)+'/data/Data.npy.npz')
        err_DoD += np.array(data['test_err'])
        err_DoD_std += np.array(data['test_err'])**2
        err_joint += np.array(data['test_err_ERM'])
        err_joint_std += np.array(data['test_err_ERM'])**2

    err_DoD_std = np.sqrt(err_DoD_std/ITER-err_DoD**2/ITER**2)
    err_CAL_std = np.sqrt(err_CAL_std/ITER-err_CAL**2/ITER**2)
    err_staged_std = np.sqrt(err_staged_std/ITER-err_staged**2/ITER**2)
    err_joint_std = np.sqrt(err_joint_std/ITER-err_joint**2/ITER**2)
    np.savez('avg.npy', err_DoD=err_DoD, err_CAL=err_CAL, err_staged=err_staged, err_joint=err_joint, err_DoD_std=err_DoD_std, err_CAL_std=err_CAL_std, err_staged_std=err_staged_std, err_joint_std=err_joint_std)


    data = np.load('avg.npy.npz')



    ax = plt.gca()
    ax.errorbar(np.arange(1, range_labels+1), data['err_DoD']/iter_size_inside_AL, yerr=data['err_DoD_std'], label="DoD",capsize=5, elinewidth=1, linewidth = 3)
    ax.errorbar(np.arange(1, range_labels+1), data['err_staged']/iter_size_inside_AL, yerr=data['err_staged_std'], label="ERM Staged", capsize=5, elinewidth=1, linewidth = 3)
    ax.errorbar(np.arange(1, range_labels+1), data['err_joint']/iter_size_inside_AL, yerr=data['err_joint_std'], label = "ERM Joint", capsize=5, elinewidth=1, linewidth = 3)
    ax.legend()
    plt.xlabel ('Size of labeled samples by human', fontsize='x-large')
    plt.ylabel ('Test error', fontsize='x-large')
    plt.grid()
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    right_side = ax.spines["top"]
    right_side.set_visible(False)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 6
    fig_size[1] = 4.2
    plt.savefig('DoD.pdf', dpi = 1000, bbox_inches='tight')

    
if __name__ == '__main__':
    run()