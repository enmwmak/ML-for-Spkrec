import pandas as pd
import numpy as np
from plda.h5helper import dict2h5, h52dict, mat2h5
import h5py as h5


# Part1 extract group info from ndx
def extract_group_info(ndx_file, group_info):
    ndx = pd.read_csv(ndx_file, sep='\t', engine='python',
                      header=0, names=['tgt', 'tst'], usecols=[0, 1])

    counts = ndx.groupby('tgt').size().values
    grp_info = []
    for count in counts:
        grp_info.append(np.repeat('grp' + str(count), count))
    grp_info = np.concatenate(grp_info)
    ndx['grp_info'] = grp_info

    ndx.set_index(['grp_info', 'tgt', 'tst'], inplace=True)

    tgt_grp = {}
    tst_grp = {}

    for lev0 in ndx.index.get_level_values(0).unique():
        tgt_grp[lev0] = ndx.loc[lev0].index.get_level_values(0).unique().values
        tst_grp[lev0] = ndx.loc[lev0].index.get_level_values(1).unique().values
    # print(tgt_grp)
    # print(tst_grp)
    dict2h5(tgt_grp, group_info, 'tgt/')
    dict2h5(tst_grp, group_info, 'tst/')


# Part2 group (only 4 group exist in ndx file) enrol and test i-vector
def group_enrol_test_ivc(group_info, enrol, test, enrol_grouped, test_group):
    with h5.File(group_info, 'r') as f:
        tgt_grp = {name: f['tgt/' + name][...] for name in f['tgt/']}
        tst_grp = {name: f['tst/' + name][...] for name in f['tst/']}

    print('Reading %s' % enrol)
    tgt = h52dict(enrol)
    print('Reading %s' % test)
    tst = h52dict(test)

    print('Saving %s' % enrol_grouped)
    with h5.File(enrol_grouped, 'w') as f:
        for grp_name, grp_spk_ids in tgt_grp.items():
            for key, val in tgt.items():
                mask = np.isin(tgt['spk_ids'], grp_spk_ids)
                f[grp_name + '/' + key] = tgt[key][mask]

    print('Saving %s' % test_group)
    with h5.File(test_group, 'w') as f:
        for grp_name, grp_spk_ids in tst_grp.items():
            for key, val in tst.items():
                mask = np.isin(tst['spk_ids'], grp_spk_ids)
                f[grp_name + '/' + key] = tst[key][mask]


