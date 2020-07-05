"""
Compute the PLDA scores according to "sre16_eval_trials.ndx". Instead of computing one score for
each line of "sre16_eval_trials.ndx", this script divides the trials into four groups according
to the number of trials for the 4 groups of target speakers. The scores_dict{} will store four matrices
with dimensions 1333x132, 2407x208, 2446x190, and 3108x272. These amount to 1986728 scores
deriving from 802 target-speakers and 9294 test utterances.
"""

import numpy as np
from plda.h5helper import h52dict_multi_level, h52dict
import pandas as pd
from collections import defaultdict
from plda.my_func import lennorm


# score_info is (802,2)-tuple. Each row corresponds to a target speaker
# whose no. of test trials is encoded in the first entry of the tuple.
# The 2nd column contains the position of the target speaker in the group
# specified in that row.
def plda_scoring(target_file, test_file, model_file, ndx_file, evl_file):
    tgt_data, tst_data, model, (ndx, score_info) = \
        load_scoring_data(target_file, test_file, model_file, ndx_file)
    tgt_data = preprocess(tgt_data, model, model['prep_mode'])
    tst_data = preprocess(tst_data, model, model['prep_mode'])
    scores = score(tgt_data, tst_data, model, score_info)
    save_evl(ndx, scores, evl_file)


def load_scoring_data(target_file, test_file, model_file, ndx_file):
    file_names = [target_file, test_file, model_file, ndx_file]
    file_types = ['target data', 'test_data', 'model', 'ndx']
    for file_type, file_name in zip(file_types, file_names):
        print('loading {0} from {1}'.format(file_type, file_name))

    return h52dict_multi_level(target_file), h52dict_multi_level(test_file), \
           h52dict(model_file), get_ndx_and_score_info(ndx_file)


def preprocess(data, model, mode):
    if mode == 'lda+lennorm':
        print(mode)
        for grp in data.keys():
            data[grp]['X'] = lennorm((data[grp]['X'] - model['mu_ind']) @ model['lda_matrix'])
        return data

    elif mode == 'ldawccn+lennorm':
        print(mode)
        for grp in data.keys():
            data[grp]['X'] = lennorm((data[grp]['X'] - model['mu_ind']) @ model['lda_wccn_matrix'])
        return data

    elif mode == 'lennorm':
        print(mode)
        for grp in data.keys():
            data[grp]['X'] = lennorm(data[grp]['X']) - model['mu2']
        return data

    elif mode == 'none':
        print(mode)
        return data
    else:
        raise NotImplementedError


# def get_ndx_and_score_info(file):
#     ndx = pd.read_csv(file, sep='[:=]', engine='python',
#                       header=None, names=['tgt', 'tst'], usecols=[0, 1])
#
#     t = ndx.groupby('tgt').count()
#     t.loc[:, 'tst'] = t.tst.apply(lambda x: 'grp' + str(x))
#     t = t.set_index('tst', append=True).swaplevel()
#     for grp_name in t.index.get_level_values(0).unique():
#         t.loc[grp_name, 'pos'] = np.arange(len(t.loc[grp_name]))
#     t.loc[:, 'pos'] = t.pos.apply(lambda x: int(x))
#     score_info = t.reset_index(level=[0, 1]).drop('tgt', axis=1).values
#
#     return ndx, score_info

def get_ndx_and_score_info(file):
    ndx_dict = h52dict(file)

    # Create Pandas DataFrame from dict. Example row of ndx: [0 1081_sre16 etabjvx_sre16_A]
    ndx = pd.DataFrame.from_records(ndx_dict)

    # How many trials for each target model. Example row of t: [1081_sre16  2407]
    t = ndx.groupby('tgt').count()

    # Append 'grp' to every row of t.tst. Example row of t: [1081_sre16  grp2407]
    t.loc[:, 'tst'] = t.tst.apply(lambda x: 'grp' + str(x))

    # Make (grp<tstcount>, <tgt>) as index. Example index in t: (grp2407, 1081_sre16)
    t = t.set_index('tst', append=True).swaplevel()

    # For each trial group (based on the no. of trials), add column 'pos' containing
    # a sequence from 0 to no_of_occurr-1. Example row of t: [grp2407 1081_sre16  0.0]
    for grp_name in t.index.get_level_values(0).unique():
        t.loc[grp_name, 'pos'] = np.arange(len(t.loc[grp_name]))

    # Change each entry in 'pos' to int
    t.loc[:, 'pos'] = t.pos.apply(lambda x: int(x))

    # Reset index at level 0 and 1. Remove column 'tgt'. Then, covert to array
    # giving ndarray containing [grg<tst_count>, no_of_occur]
    score_info = t.reset_index(level=[0, 1]).drop('tgt', axis=1).values
    return ndx, score_info


def score(tgt_data, tst_data, model, score_info):
    tgt = defaultdict(dict)
    for grp in tgt_data.keys():
        X = tgt_data[grp]['X']
        spk_ids = tgt_data[grp]['spk_ids']
        xQxt_mat, X_avg = _comp_tgt_xQxt_mat(X, spk_ids, model['Q'])
        tgt[grp]['xQxt_mat'] = xQxt_mat
        tgt[grp]['X_avg'] = X_avg

    tst = defaultdict(dict)
    for grp in tst_data.keys():
        X = tst_data[grp]['X']
        QX = X @ model['Q']
        xQxt_mat = np.einsum('ij,ij->i', X, QX)     # Diag(X * Q * X') for row-vectors in X
        # xQxt_mat = np.diag(X @ QX.T)
        tst[grp]['xQxt_mat'] = xQxt_mat
        tst[grp]['X'] = X

    scores_dict = {}
    for grp in tgt.keys():
        X1tP = model['P'] @ tgt[grp]['X_avg'].T
        X1tPX2 = tst[grp]['X'] @ X1tP               # Compute the terms (x_s'*P*x_t) in the scoring function
        scores_dict[grp] = (
            0.5 * tgt[grp]['xQxt_mat']              # Repeat rows to make n_tst rows, e.g., (1332,132)
            + 0.5 * tst[grp]['xQxt_mat'][:, None]   # Repeat columns to make n_tgt columns
            + X1tPX2 + model['const']               # matrix with dim n_tst x n_tgt, e.g., (1332,132)
        )

    # For each row in score_info (each represent one target speaker),
    # extract the scores of the trials corresponding to that target speaker.
    # The variable "index" is to index the column in score_dict[grp] corresponding
    # to that target speaker.
    scores = []
    for grp, index in score_info:
        scores.append(scores_dict[grp][:, index])

    # Concatenate the scores in the score list (scores) and convert it to ndarray
    return np.concatenate(scores)


def _comp_tgt_xQxt_mat(X, spk_ids, Q):
    xQxt_mat = []
    X_avg = []

    # Compute diag(XQX^T) for row-vectors in X, which is equivalent
    # to diag(X^T * Q * X) for column-vectors in X
    QX = X @ Q                              # Note: X contains row vectors instead of column vectors
    xQxt_raw = np.einsum('ij,ij->i', X, QX) # Compute diag(X * B^T), where B is the var QX
    # xQxt_raw = np.diag(X @ QX.T)

    # Get the indexes to the first occurence of the sorted spk_ids
    _, idx = np.unique(spk_ids, return_index=True)

    # Get the first occurrence of each spkID in the ndarry spk_ids
    uni_spk_ids = spk_ids[np.sort(idx)]

    # For each unique spk_id, compute the mean of his/her x'Qx and i-vec mean
    for uni_spk_id in uni_spk_ids:
        mask = spk_ids == uni_spk_id
        xQxt_mat.append(xQxt_raw[mask].mean(0))
        X_avg.append(X[mask].mean(0))

    # convert lists into ndarrays and return
    return np.stack(xQxt_mat), np.stack(X_avg)


def save_evl(ndx, scores, evl_file):
    ndx.columns = ['modelid', 'segmentid']
    df = ndx
    df['LLR'] = scores.astype('U10')
    df_expand = df['segmentid'].str.split(r'_(?=[AB])', expand=True)  # Split testutt name and channel
    df['segmentid'] = df_expand[0]                                    # Keep the testutt name only
    #df['side'] = df_expand[1].str.lower()                           # Side means the channel in lower case
    df['side'] = 'a'                                                # Always channel 'a' in SRE18
    df.to_csv(evl_file, header=True, index=False, sep='\t',
              columns=['modelid', 'segmentid', 'side', 'LLR'])
    print('save evl to {}'.format(evl_file))


# for key, val in scores_dict.items():
#     print(val.shape)

# plda_scoring(
#     target_file='../data/h5/grouped/sre16_eval_enrollments_t300_w_512c.h5',
#     test_file='../data/h5/grouped/sre16_eval_tstutt_t300_w_512c.h5',
#     model_file='../data/h5/model/semi_plda.h5',
#     ndx_file='../lst/sre16_eval_trials_ndx.lst',
#     # ndx_file='../lst/ndx.h5',
#     evl_file='../evl/junk.evl',
#     prep_mode='lennorm'
# )
