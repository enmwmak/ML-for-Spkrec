import pandas as pd
import h5py as h5
import numpy as np
import scipy.io as sio
from collections import defaultdict
import os

unicode = h5.special_dtype(vlen=str)


def h52dict_multi_level(file):
    with h5.File(file, 'r') as f:
        dict_out = defaultdict(dict)
        for grp in f:
            for field in f[grp]:
                dict_out[grp][field] = f[grp][field][...]
    return dict_out


def h52dict(file, n_data=None):
    with h5.File(file, 'r') as f:
        if n_data is None:
            return {name: f[name][...] for name in f}
        else:
            return {name: f[name][...][:n_data] for name in f}


def dict2h5(in_dict, file, dataset=''):
    with h5.File(file, 'a') as f:
        for key, val in in_dict.items():
            if val.dtype == np.object:
                f[dataset + key] = val.astype(unicode)
            else:
                f[dataset + key] = val


def concath5(read_file_names, write_file_name):
    # check all fields are the same
    fields = []
    for file_name in read_file_names:
        with h5.File(file_name, 'r') as f:
            fields.append(sorted(list(f)))
    fields = np.array(fields)
    if not np.all(fields == fields[0]):
        raise ValueError('fields are not consistent in input files')
    print(fields)
    data = {name: [] for name in fields[0]}

    for file_name in read_file_names:
        with h5.File(file_name, 'r') as f:
            for name in f:
                data[name].append(f[name][...])

    with h5.File(write_file_name, 'a') as f:
        for key, val in data.items():
            if val[0].dtype == np.object:
                f[key] = np.concatenate(val, axis=0).astype(unicode)
                print(key)
            else:
                f[key] = np.concatenate(val, axis=0)

# read_file_names = ['../data/h5/domain/female_plda_nist_t300_w_512c.h5',
#                     '../data/h5/domain/male_plda_nist_t300_w_512c.h5',
#  ]
# write_file_name = '../junk/all_plda_nist_t300_w_512c.h5'
# concath5(read_file_names, write_file_name)


def mat2h5(mat_file, h5_file,  n_dims_keeped=None):
    with h5.File(h5_file, 'a') as f:
        mat_conts = sio.loadmat(mat_file, squeeze_me=True)
        if n_dims_keeped is not None:
            mat_conts['w'] = mat_conts['w'][:, :n_dims_keeped]
        f['spk_ids'] = mat_conts['spk_logical'].astype(unicode)
        f['X'] = mat_conts['w']
        f['n_frames'] = mat_conts['num_frames']
        f['spk_path'] = mat_conts['spk_physical'].astype(unicode)


# Part 3 convert ndx text file in to h5 to speed up loading
#  (you can skip this part it only speeds up about 15 sec, but if so remember to change ndx loading function)
def ndx2h5(ndx_txt, ndx_h5):
    ndx = pd.read_csv(ndx_txt, sep='[:=]', engine='python',
                      header=None, names=['tgt', 'tst'], usecols=[0, 1])
    dict2h5(in_dict={'tgt': ndx.tgt.values, 'tst': ndx.tst.values,}, file=ndx_h5)


# Convert trial file (in .tsv format) to .h5 file
def trial2h5(trial_file, h5_file):
    ndx = pd.read_csv(trial_file, sep='\t', engine='python',
                      header=0, names=['tgt', 'tst'], usecols=[0, 1])
    dict2h5(in_dict={'tgt': ndx.tgt.values, 'tst': ndx.tst.values,}, file=h5_file)


# Run this under the lib/ folder
if __name__ == '__main__':

    ndxfile = '/home5a/mwmak/so/spkver/callmynet/matlab/ivec/lst/fw60/sre16_dev_trials_ndx.lst'
    h5file = '../data/h5/sre16_dev_trials_ndx.h5'
    try:
        os.remove(h5file)
    except OSError:
        pass
    print('{0} --> {1}'.format(ndxfile, h5file))
    ndx2h5(ndxfile, h5file)

    matfiles = [
                '../../matlab/ivec1/mat/fw60/male_plda_combine_t300_w_512c-1.mat',
                '../../matlab/ivec1/mat/fw60/male_plda_combine_t300_w_512c-2.mat',
                '../../matlab/ivec1/mat/fw60/male_plda_combine_t300_w_512c-3.mat',
                '../../matlab/ivec1/mat/fw60/male_plda_combine_t300_w_512c-4.mat',
                '../../matlab/ivec1/mat/fw60/female_plda_combine_t300_w_512c-1.mat',
                '../../matlab/ivec1/mat/fw60/female_plda_combine_t300_w_512c-2.mat',
                '../../matlab/ivec1/mat/fw60/female_plda_combine_t300_w_512c-3.mat',
                '../../matlab/ivec1/mat/fw60/female_plda_combine_t300_w_512c-4.mat',
                '/home5a/mwmak/so/spkver/callmynet/matlab/ivec/mat/fw60/sre16_dev_enrollments_t300_w_512c.mat',
                '/home5a/mwmak/so/spkver/callmynet/matlab/ivec/mat/fw60/sre16_dev_tstutt_t300_w_512c.mat',
                '../../matlab/ivec1/mat/fw60/gmix_plda_sre16_dev_t300_w_512c.mat',
                '../../matlab/ivec1/mat/fw60/gmix_plda_sre16_eval_t300_w_512c.mat',
                '../../matlab/ivec1/mat/fw60/male_ceb_plda_sre16_dev_t300_w_512c.mat',
                '../../matlab/ivec1/mat/fw60/male_cmn_plda_sre16_dev_t300_w_512c.mat',
                '../../matlab/ivec1/mat/fw60/female_ceb_plda_sre16_dev_t300_w_512c.mat',
                '../../matlab/ivec1/mat/fw60/female_cmn_plda_sre16_dev_t300_w_512c.mat',
                '../../matlab/ivec1/mat/fw60/male_tgl_plda_sre16_eval_t300_w_512c.mat',
                '../../matlab/ivec1/mat/fw60/male_yue_plda_sre16_eval_t300_w_512c.mat',
                '../../matlab/ivec1/mat/fw60/female_tgl_plda_sre16_eval_t300_w_512c.mat',
                '../../matlab/ivec1/mat/fw60/female_yue_plda_sre16_eval_t300_w_512c.mat',
                ]
    h5files = [
                '../data/h5/male_plda_combine_t300_w_512c-1.h5',
                '../data/h5/male_plda_combine_t300_w_512c-2.h5',
                '../data/h5/male_plda_combine_t300_w_512c-3.h5',
                '../data/h5/male_plda_combine_t300_w_512c-4.h5',
                '../data/h5/female_plda_combine_t300_w_512c-1.h5',
                '../data/h5/female_plda_combine_t300_w_512c-2.h5',
                '../data/h5/female_plda_combine_t300_w_512c-3.h5',
                '../data/h5/female_plda_combine_t300_w_512c-4.h5',
                '../data/h5/sre16_dev_enrollments_t300_w_512c.h5',
                '../data/h5/sre16_dev_tstutt_t300_w_512c.h5',
                '../data/h5/gmix_plda_sre16_dev_t300_w_512c.h5',
                '../data/h5/gmix_plda_sre16_eval_t300_w_512c.h5',
                '../data/h5/male_ceb_plda_sre16_dev_t300_w_512c.h5',
                '../data/h5/male_cmn_plda_sre16_dev_t300_w_512c.h5',
                '../data/h5/female_ceb_plda_sre16_dev_t300_w_512c.h5',
                '../data/h5/female_cmn_plda_sre16_dev_t300_w_512c.h5',
                '../data/h5/male_tgl_plda_sre16_eval_t300_w_512c.h5',
                '../data/h5/male_yue_plda_sre16_eval_t300_w_512c.h5',
                '../data/h5/female_tgl_plda_sre16_eval_t300_w_512c.h5',
                '../data/h5/female_yue_plda_sre16_eval_t300_w_512c.h5'
            ]
    for mfile, hfile in zip(matfiles, h5files):
        try:
            os.remove(hfile)
        except OSError:
            pass
        print('{0} --> {1}'.format(mfile, hfile))
        mat2h5(mfile, hfile)

