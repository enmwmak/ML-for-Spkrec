"""
Convert .h5 file to .ark and .scp file. The .h5 file should have two fields ('spk_ids' and 'X')
representing the Speaker ID and i/x-vectors. 
The filepath in the .scp file is the same as the 2nd parameter of the function h52ark_scp().
"""

import kaldi_io
import pathlib
import h5py


def h52dict(file, n_data=None):
    with h5py.File(file, 'r') as f:
        if n_data is None:
            return {name: f[name][...] for name in f}
        else:
            return {name: f[name][...][:n_data] for name in f}


def h52ark_scp(in_name, out_name):
    pathlib.Path('/'.join(out_name.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    data = h52dict(in_name)
    new_data = {utt_id: ivector for utt_id, ivector in zip(data['spk_ids'], data['X'])}
    ark_scp_output = 'ark:| /usr/local/kaldi/src/bin/copy-vector ark:- ark,scp:{0}.ark,{0}.scp'.format(
                        out_name)
    with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f:
        for key, mat in new_data.items():
            kaldi_io.write_vec_flt(f, mat, key=key)


if __name__ == '__main__':
    h52ark_scp('data/h5/ivectors_sre18_dev_enroll.h5', 'data/exp/ivectors_sre18_dev_enroll/xvector')