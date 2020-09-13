import h5py as h5
import numpy as np
# import cupy as np
unicode = h5.special_dtype(vlen=str)
def h52dict(file, n_data=None):
    with h5.File(file, 'r') as f:
        if n_data is None:
            return {name: f[name][...] for name in f}
        else:
            return {name: f[name][...][:n_data] for name in f}

def dict2h5(in_dict, file, dataset=''):
    with h5.File(file, 'w') as f:
        for key, val in in_dict.items():
            if val.dtype == np.object:
                f[dataset + key] = val.astype(unicode)
            else:
                f[dataset + key] = val
        f.close()