from plda.preprocess import PreProcess
import pandas as pd
import h5py as h5
import numpy as np
import warnings


def plda_scoring(target_file, test_file, model_file, ndx_file, evl_file):
    tgt, tst, ndx, model = load_scoring_data(target_file, test_file, model_file, ndx_file)
    tgt['X'] = preprocess(tgt['X'], model, model['prep_mode'])
    tst['X'] = preprocess(tst['X'], model, model['prep_mode'])
    scores = scoring(tgt, tst, ndx, model)
    save_evl(ndx, scores, evl_file)


def load_scoring_data(target_file, test_file, model_file, ndx_file):
    with h5.File(target_file, 'r') as f:
        print('loading target data from {}'.format(target_file))
        tgt = {name: f[name][...] for name in f}

    with h5.File(test_file, 'r') as f:
        print('loading test data from {}'.format(test_file))
        tst = {name: f[name][...] for name in f}

    with h5.File(model_file, 'r') as f:
        print('loading model from {}'.format(model_file))
        model = {name: f[name][...] for name in f}

    ndx = pd.read_csv(ndx_file, dtype=np.str, sep=r'[=:]',
                      header=None, usecols=[0, 1], engine='python').values
    print('loading ndx from {}'.format(ndx_file))
    return tgt, tst, ndx, model


def preprocess(X, model, mode):
    prep = PreProcess(X, model=model)
    if mode == 'wccn+lennorm+lda+wccn':
        print(mode)
        return prep.trans_wccn().demean('mu1').\
            lennorm().trans_lda_wccn().demean('mu2').X
    elif mode == 'center+whiten+lennorm':
        print(mode)
        return prep.demean('mu1').trans_whiten().lennorm().demean('mu2').X
    elif mode == 'lennorm':
        print(mode)
        return prep.lennorm().demean('mu2').X
    elif mode == 'wccn+lennorm':
        print(mode)
        return prep.trans_wccn().lennorm().demean('mu2').X
    elif mode == 'whiten+lennorm':
        print(mode)
        return prep.trans_whiten().lennorm().demean('mu2').X
    else:
        raise NotImplementedError


def scoring(tgt, tst, ndx, model):
    # warnings.warn('reduce to single precision')
    # tgt['X'] = tgt['X'].astype(np.float32).copy()
    # tst['X'] = tst['X'].astype(np.float32).copy()
    # model['P'] = model['P'].astype(np.float32).copy()
    # model['Q'] = model['Q'].astype(np.float32).copy()
    # model['const'] = model['const'].astype(np.float32).copy()
    # scores = np.zeros(len(ndx), dtype=np.float32)

    scores = np.zeros(len(ndx))
    tgt_dict = _build_id_dict(tgt['X'], tgt['spk_ids'])
    tst_dict = _build_id_dict(tst['X'], tst['spk_ids'])
    for i, (tgt_id, tst_name) in enumerate(ndx):
        scores[i] = _comp_llikelh_ratio(
            X=tgt_dict[tgt_id],
            y=tst_dict[tst_name].squeeze(),
            P=model['P'],
            Q=model['Q'],
            const=model['const']
        )
        if i % 100000 == 0:
            print('{}/{},{},{},{}'.format(
                i, scores.shape[0], tgt_id, tst_name, str(scores[i]),))
    return scores


# def _scoring(i, tgt_id, tst_name):
#     scores = _comp_llikelh_ratio(
#         X=tgt_dict[tgt_id],
#         y=tst_dict[tst_name].squeeze(),
#         P=model['P'],
#         Q=model['Q'],
#         const=model['const']
#     )
#     if i % 100000 == 0:
#         print('{},{},{},{}'.format(
#             i, tgt_id, tst_name, str(scores),))


def _build_id_dict(X, spk_ids):
    return {spk_id: X[spk_ids == spk_id] for spk_id in np.unique(spk_ids)}


def _comp_llikelh_ratio(X, y, P, Q, const):
    return (
        0.5 * ravel_dot(X.T @ X, Q) / X.shape[0]
        + kernel_dot(y, P, X.sum(0).T) / X.shape[0]
        + 0.5 * kernel_dot(y, Q, y.T)
        + const
    )


def ravel_dot(X, Y):
    return X.ravel() @ Y.ravel()


def kernel_dot(x, kernel, y):
    return x @ kernel @ y


def save_evl(ndx, scores, evl_file):
    df = pd.DataFrame(ndx, columns=['modelid', 'segment'])
    df['LLR'] = scores.astype('U10')
    df_expand = df['segment'].str.split(r'_(?=[AB])', expand=True)
    df['segment'] = df_expand[0]
    df['side'] = df_expand[1].str.lower()
    df.to_csv(evl_file, header=True, index=False, sep='\t',
              columns=['modelid', 'segment', 'side', 'LLR'])
    print('save evl to {}'.format(evl_file))




