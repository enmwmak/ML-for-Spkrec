"""
Utility functions supporting xvec_adv_emb.py, xvec_dnn_cls.py, and xvec_adv_tx.py
M.W. Mak, Oct. 2018
"""

from __future__ import print_function

import numpy as np
from keras.utils import np_utils
from kaldi_arkread import load_scpfile
from itertools import chain
import h5py


def get_accuracy(y_pred, y_true):
    n_correct = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1), axis=0)    
    return 100*n_correct / float(y_pred.shape[0])


# Load i/x-vectors from .ark file through .scp file. Return i/x-vectors, speaker lables 
# (start from 0), domain labels, and utterance names
def load_xvectors(domains, expdir, vectype, datadir, basedir, min_n_vecs=20):
    spk2uttmap = dict()
    utt2vecmap = dict()
    utt2dommap = dict()
    d = 0
    for dom in domains:
        scpfile = expdir + vectype + 's_' + dom + '/' + vectype + '.scp'
        spk2uttfile = datadir + dom + '/spk2utt'
        spk2utt = load_spk2utt(spk2uttfile)
        #spk2uttmap = {**spk2uttmap, **spk2utt}        # Concate dict. Keys in dicts should be mutually exclusive
        spk2uttmap = merge_dicts(spk2uttmap, spk2utt)  # Merge dict. Some keys in dicts could be the same
        [key, val] = load_scpfile(basedir, scpfile, arktype='vec')
        utt2vec = dict(zip(key, val))
        utt2vecmap = {**utt2vecmap, **utt2vec}
        utt2dom = dict(zip(key, [d]*len(key)))
        utt2dommap = {**utt2dommap, **utt2dom}
        d = d + 1
    print('Preparing data')
    xvecs, spk_lbs, dom_lbs, utt_ids = prepare_data(spk2uttmap, utt2vecmap, utt2dommap)
    print('Extracting data')
    xvecs, spk_lbs, dom_lbs, utt_ids = extract_data(xvecs, spk_lbs, dom_lbs, utt_ids, min_n_vecs=min_n_vecs, 
                                            n_spks=-1, shuffle=False)
    return xvecs, spk_lbs, dom_lbs, utt_ids


# Load i/x-vectors, spk labels, domain labels, utt_ids from .h5 file
def load_xvectors_h5(h5file):
    print('Loading %s' % h5file)
    with h5py.File(h5file, 'r') as f:
        X = f['X'][:]
        spk_lbs = f['spk_lbs'][:]
        dom_lbs = f['dom_lbs'][:]
        utt_ids = f['utt_ids'][:]
    return X, spk_lbs, dom_lbs, utt_ids    

# Save i/x-vectors, spk labels, domain labels, utt_ids to .h5 file
def save_xvectors_h5(h5file, X, spk_lbs, dom_lbs, utt_ids, domains=['unknown']):
    unicode = h5py.special_dtype(vlen=str)
    print('Saving %s' % h5file)
    with h5py.File(h5file, 'w') as f:
        f['X'] = X
        f['spk_lbs'] = spk_lbs
        f['dom_lbs'] = dom_lbs
        f['utt_ids'] = utt_ids.astype(unicode)
        f['domains'] = np.array(domains).astype(unicode)


# Merge two dictionaries, concatenating the values of the same key in the two dicts.
def merge_dicts(dict1, dict2):
    dict3 = dict()
    for k, v in chain(dict1.items(), dict2.items()):
        if k in dict3:
            dict3[k].extend(v)
        else:
            dict3[k] = v    
    return dict3    


def split_trn_tst(x, spk_lbs, dom_lbs, utt_ids, ratio=0.5):
    x_trn = list()
    spk_lbs_trn = list()
    dom_lbs_trn = list()
    utt_ids_trn = list()
    x_tst = list()
    spk_lbs_tst = list()
    dom_lbs_tst = list()
    utt_ids_tst = list()
    for s in np.unique(spk_lbs):
        idx = [i for i, e in enumerate(spk_lbs) if e == s]
        trn_idx = idx[0:int(ratio*len(idx))]
        tst_idx = idx[int(ratio*len(idx))+1:]
        x_trn.append(x[trn_idx,:])
        spk_lbs_trn.extend(list(spk_lbs[trn_idx]))
        utt_ids_trn.extend(list(utt_ids[trn_idx]))
        dom_lbs_trn.extend(list(dom_lbs[trn_idx]))
        x_tst.append(x[tst_idx,:])
        spk_lbs_tst.extend(list(spk_lbs[tst_idx]))
        utt_ids_tst.extend(list(utt_ids[tst_idx]))
        dom_lbs_tst.extend(list(dom_lbs[tst_idx]))
    x_trn = np.vstack(x_trn)
    spk_lbs_trn = np.array(spk_lbs_trn)
    utt_ids_trn = np.array(utt_ids_trn)
    dom_lbs_trn = np.array(dom_lbs_trn)
    x_tst = np.vstack(x_tst)
    spk_lbs_tst = np.array(spk_lbs_tst)
    utt_ids_tst = np.array(utt_ids_tst)
    dom_lbs_tst = np.array(dom_lbs_tst)
    return x_trn, spk_lbs_trn, dom_lbs_trn, utt_ids_trn, x_tst, spk_lbs_tst, dom_lbs_tst, utt_ids_tst


def select_speakers(x, spk_lbs, dom_lbs, utt_ids, min_n_vecs=20, n_spks=10):
    # Return a data matrix containing n_spks from each domain. 
    # Also return the speaker labels and domain labels 
    sel_x = list()
    sel_spk_lbs = list()
    sel_dom_lbs = list()
    sel_utt_ids = list()
    for d in range(np.max(dom_lbs) + 1):
        idx = [i for i, e in enumerate(dom_lbs) if e == d]
        s_x, s_spk_lbs, s_dom_lbs, s_utt_ids = extract_data(x[idx, :], spk_lbs[idx], dom_lbs[idx], utt_ids[idx],
                                                    min_n_vecs=min_n_vecs, n_spks=n_spks, shuffle=False)
        sel_x.append(s_x)
        sel_dom_lbs.extend(list(s_dom_lbs))
        sel_utt_ids.extend(list(s_utt_ids))
        if (d > 0):
            s_spk_lbs = s_spk_lbs + np.max(sel_spk_lbs) + 1         # extract_data returns lbs starts from 0
        sel_spk_lbs.extend(list(s_spk_lbs))
    return np.vstack(sel_x), np.array(sel_spk_lbs), np.array(sel_dom_lbs), np.array(sel_utt_ids)


def prepare_data(spk2uttmap, utt2vecmap, utt2dommap):
    # Return a data matrix and its speaker labels and domain labels (all are ndarray)
    spk_lbs = list()
    dom_lbs = list()
    xvectors = list()
    utt_ids = list()
    spk_no = 0
    for spkid in spk2uttmap:
        utts = spk2uttmap[spkid]
        for u in utts:
            if u in utt2vecmap and u in utt2dommap:
                xvectors.append(utt2vecmap[u])
                spk_lbs.append(spk_no)
                dom_lbs.append(utt2dommap[u])
                utt_ids.append(u)
            else:
                print('Warning: Utterance %s not in xvector.scp' % u)
        spk_no = spk_no + 1
    xvectors = np.vstack(xvectors)    
    return xvectors, np.array(spk_lbs), np.array(dom_lbs), np.array(utt_ids)


def load_spk2utt(spk2uttfile):
    # Read Kaldi's spk2utt file and return dict {spkid: [utt_id1,utt_id2,...]}
    spk2utt = dict()
    with open(spk2uttfile) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]      # Strip '\n'
    for line in lines:
        tokens = line.split()
        key = tokens.pop(0)
        spk2utt[key] = tokens
    return spk2utt    


def extract_data(x, spk_lbs, dom_lbs, utt_ids, min_n_vecs, n_spks=-1, shuffle=False):
    # Returen data matrix, speaker labels and domain labels with no. of utts per speaker >= min_n_vecs
    idx_lst = list()
    new_spk_lbs = list()
    unq_spk_lbs = np.unique(np.asarray(spk_lbs))        # Find a list of unique speaker labels
    count = 0                                           # New spk labels start from 0
    if n_spks == -1:
        n_spks = len(unq_spk_lbs)                       # Default is to extract the data of all speakers
    for lbs in unq_spk_lbs:
        idx = [t for t, e in enumerate(spk_lbs) if e == lbs]  # Find idx in spk_trn that match spkid
        if len(idx) >= min_n_vecs:
            idx_lst.extend(idx)
            new_spk_lbs.extend([count] * len(idx))      # Repeat count len(idx) times. Assign new labels
            count = count + 1
            if count == n_spks:
                break

    new_spk_lbs = np.array(new_spk_lbs)
    new_x = x[idx_lst, :]
    new_dom_lbs = dom_lbs[idx_lst]
    new_utt_ids = utt_ids[idx_lst]
    if shuffle:
        idx = np.random.permutation(len(new_spk_lbs))
        new_x = new_x[idx, :]
        new_spk_lbs = new_spk_lbs[idx]
        new_dom_lbs = new_dom_lbs[idx]
        new_utt_ids = new_utt_ids[idx]

    return new_x, new_spk_lbs, new_dom_lbs, new_utt_ids



