import h5py as h5
from scipy.io import loadmat, savemat
import numpy as np
# import cupy as np
from utili_ import dict2h5
from bosaris_toolkit.key import Key
from bosaris_toolkit.scores import Scores
from bosaris_toolkit.detplot import effective_prior
from bosaris_toolkit.detplot import rocch
from bosaris_toolkit.detplot import rocch2eer
from bosaris_toolkit.detplot import sigmoid

class Results:

    def __init__(self):
        self.tar = np.empty(0, dtype=np.float64)
        self.non = np.empty(0, dtype=np.float64)
        self.knnon = np.empty(0, dtype=np.float64)
        self.uknnon = np.empty(0, dtype=np.float64)
        self.Pmiss = np.empty(0, dtype=np.float64)
        self.Pfa = np.empty(0, dtype=np.float64)
        self.Pfa_kn = np.empty(0, dtype=np.float64)
        self.Pfa_ukn = np.empty(0, dtype=np.float64)
        self.thresholds = np.empty(0, dtype=np.float64)


def full_roc(tar_scores, nontar_scores, knnontar_scores, uknnontar_scores, nThresholds=20000):
    """
     Compute the full ROC by sweeping nThresholds ranging from
     the smallest score and the largest scores in tar_scores, and nontar_scores
     Author: M.W. Mak
    """

    score = np.hstack((tar_scores.T, nontar_scores.T))
    dx = (np.max(score) - np.min(score)) / nThresholds
    assert dx > 0, 'The two score distributions do not overlap\n'
    maxs = np.max(score) - dx
    mins = np.min(score) + dx
    X = np.array(range(nThresholds) * (maxs - mins) / (nThresholds - 1) + mins)

    # Compute FP and FN at different thresholds
    fn = np.zeros(shape=(len(X),), dtype=np.float64)
    fp = np.zeros(shape=(len(X),), dtype=np.float64)
    for i in range(len(X)):
        fn[i] = len(np.where(tar_scores <= X[i])[0])
        fp[i] = len(np.where(nontar_scores > X[i])[0])

    # Compute FAR and FRR at different thresholds
    Pmiss = fn / len(tar_scores)
    Pfa = fp / len(nontar_scores)

    for i in range(len(Pfa)):
        if Pmiss[i] >= Pfa[i]:
            break
    eer = (Pmiss[i] + Pfa[i]) / 2

    # Compute the Pfa of known and unknown nontargets
    for i in range(len(X)):
        fp[i] = len(np.where(knnontar_scores > X[i])[0])
    Pfa_kn = fp / len(knnontar_scores)
    for i in range(len(X)):
        fp[i] = len(np.where(uknnontar_scores > X[i])[0])

    # To prevent the case where there is no unknown non-target (e.g., CC4 male))
    if uknnontar_scores.size == 0:
        Pfa_ukn = fp
    else:
        Pfa_ukn = fp / len(uknnontar_scores)

    return Pmiss, Pfa, Pfa_kn, Pfa_ukn, eer, X

def get_norm_min_dcf(res, prior1, prior2):

    Pknown = 0.5
    Ptar1 = prior1
    Pnon1 = 1 - prior1
    Ptar2 = prior2
    Pnon2 = 1 - prior2
    res.Pmiss, res.Pfa, res.Pfa_kn, res.Pfa_ukn, _, res.thresholds = full_roc(res.tar, res.non, res.knnon, res.uknnon)
    cdet1 = np.dot(np.hstack((Ptar1, Pnon1)), np.vstack((res.Pmiss.T, Pknown * res.Pfa_kn.T + (1 - Pknown) * res.Pfa_ukn.T)))
    cdet2 = np.dot(np.hstack((Ptar2, Pnon2)), np.vstack((res.Pmiss.T, Pknown * res.Pfa_kn.T + (1 - Pknown) * res.Pfa_ukn.T)))
    ncdet1 = cdet1 / np.min((Ptar1, Pnon1))
    ncdet2 = cdet2 / np.min((Ptar2, Pnon2))
    mindcf = (np.min((np.min(ncdet1), 2)) + np.min((np.min(ncdet2), 2))) / 2
    return mindcf

def fast_sre12_actDCF(tar, knnon, uknnon, plo, normalize = False):
    """
         Modified by Mak based on fast_actDCF.m for SRE12 actual DCF
     Computes the actual average cost of making Bayes decisions with scores
     calibrated to act as log-likelihood-ratios. The average cost (DCF) is
     computed for a given range of target priors and for unity cost of error.
     If un-normalized, DCF is just the Bayes error-rate.

      Usage examples:  dcf = fast_actDCF(tar,non,-10:0.01:0)
                       norm_dcf = fast_actDCF(tar,non,-10:0.01:0,true)
                       [dcf,pmiss,pfa] = fast_actDCF(tar,non,-10:0.01:0)

      Inputs:
        tar: a vector of T calibrated target scores
        non: a vector of N calibrated non-target scores
             Both are assumed to be of the form

                   log P(data | target)
           llr = -----------------------
                 log P(data | non-target)

             where log is the natural logarithm.
        knnon: a vector of N_kn calibrated known-nontarget scores
        uknnon: a vector of N_ukn calibrated unknown-nontarget scores

        plo1 and plo2: an ascending vector of log-prior-odds, plo = logit(Ptar)
                                                        = log(Ptar) - log(1-Ptar)

        normalize: (optional, default false) return normalized dcf if true.


       Outputs:
         dcf: a vector of DCF values, one for every value of plo.

                 dcf(plo) = Ptar(plo)*Pmiss(plo) + (1-Ptar(plo1))*Pfa(plo)

              where Ptar(plo) = sigmoid(plo) = 1./(1+exp(-plo)) and
              where Pmiss and Pfa are computed by counting miss and false-alarm
              rates, when comparing 'tar' and 'non' scores to the Bayes decision
              threshold, which is just -plo. If 'normalize' is true, then dcf is
              normalized by dividing by min(Ptar,1-Ptar).

          Pmiss: empirical actual miss rate, one value per element of plo.
                 Pmiss is not altered by parameter 'normalize'.

          Pfa: empirical actual false-alarm rate, one value per element of plo.
               Pfa is not altered by parameter 'normalize'.

     Note, the decision rule applied here is to accept if

        llr >= Bayes threshold.

     or reject otherwise. The >= is a consequence of the stability of the
     sort algorithm , where equal values remain in the original order.
    """

    assert len(tar.shape) <= 1
    assert len(knnon.shape) <= 1
    assert len(uknnon.shape) <= 1
    assert len(plo.shape) <= 1

    assert (np.array([plo]) == np.sort(np.array([plo]))).any(), 'Parameter plo1 must be in ascending order.'

    D = plo.size
    T = tar.size
    N_kn = knnon.size
    N_ukn = uknnon.size

    tar = tar.reshape((1, -1))
    knnon = knnon.reshape((1, -1))
    plo = plo.reshape((1, -1))
    uknnon = uknnon.reshape((1, -1))

    ii = np.argsort(np.hstack((-plo, tar)))[0]  # -plo are thresholds
    r = np.zeros(shape=(T + D), dtype=np.float64)
    r[ii] = np.array(range(1, T+D+1))
    r = r[0:D]  # rank of thresholds
    Pmiss = r - np.array(range(D, 0, -1))

    ii = np.argsort(np.hstack((-plo, knnon)))[0]  # -plo are thresholds
    r = np.zeros(shape=(N_kn+D), dtype=np.float64)
    r[ii] = np.array(range(1, N_kn+D+1))
    r = r[0:D]   # rank of thresholds
    Pfa_kn = N_kn - r + np.array(range(D, 0, -1))

    ii = np.argsort(np.hstack((-plo, uknnon)))[0]  # -plo are thresholds
    r = np.zeros(shape=(N_ukn+D), dtype=np.float64)
    r[ii] = np.array(range(1, N_ukn+D+1))
    r = r[0:D]   # rank of thresholds
    Pfa_ukn = N_ukn - r + np.array(range(D, 0, -1))

    Pmiss = Pmiss / T
    Pfa_kn = Pfa_kn / N_kn

    if N_ukn > 0:   # Prevent divided by 0 in case there is on unknown non-targets
        Pfa_ukn = Pfa_ukn / N_ukn

    Ptar = sigmoid(plo)
    Pnon = sigmoid(-plo)
    Pknown = 0.5

    dcf = Ptar * Pmiss + Pnon * (np.dot(Pknown, Pfa_kn) + np.dot((1 - Pknown), Pfa_ukn))

    if normalize:
        dcf = dcf / np.min((Ptar, Pnon))

    return dcf[0], Pmiss, Pfa_kn, Pfa_ukn

def logit(temp):
    return np.log(temp / (1 - temp))

def get_norm_act_dcf(res, prior1, prior2):
    actdcf1, _, _, _ = fast_sre12_actDCF(res.tar, res.knnon, res.uknnon, logit(prior1), normalize=True)
    actdcf2, _, _, _ = fast_sre12_actDCF(res.tar, res.knnon, res.uknnon, logit(prior2), normalize=True)
    actdcf = (actdcf1 + actdcf2) / 2
    return actdcf

def eval_by_bosaris(sre12_evlfile, sre12_keyfile = './key/NIST_SRE12_core_trial_key.v1', fast_keyloading = True):
    """
     Use Bosaris toolkit for evaluation.
     be
     Input:
       sre12_evlfile      - Evaluation file that match the segment name of key file
       keyfile            - Optional key file (work for SRE12 key file)
       fast_keyloading    - Optinoal flag for fast keyfile loading (load a pre-stored .h5 key file)
     Output:
       eer                - Equal error rate
       dcf12              - Minimum DCF as defined in SRE12
       res12              - Result12 object in Bosaris toolkit
     Example:
       evl2evl('evl/fw60/gplda60_male_cc4_1024c.evl', 'ndx/male/core-core_8k_male_cc4.ndx','evl/fw60/sre12_gplda60_male_cc4_1024c.evl');
       [eer,dcf12,res12] = eval_by_bosaris('evl/fw60/sre12_gplda60_male_cc4_1024c.evl','../../key/NIST_SRE12_core_trial_key.v1',true);

     Note that user need to run scripts/re_arrange_evl.pl or evl2evl_.py to produce an .evl file that can match the
     segment name of the key file.

     Set to true if .h5 format of the key file has been saved from previous run

    """

    sre12_keyfile_h5_path = sre12_keyfile + '.h5'

    if not fast_keyloading:

        temp = loadmat(sre12_keyfile + '.mat')['key'][0][0]
        key12 = {'modelset': np.array([item[0][0].encode() for item in temp[0]]),\
                 'segset': np.array([item[0][0].encode()for item in temp[1]]),\
                 'tar': temp[2],\
                 'non': temp[3],\
                 'knnon': temp[4], \
                 'uknnon': temp[5]}

        dict2h5(key12, file=sre12_keyfile_h5_path)

    key12 = Key(key_file_name=sre12_keyfile_h5_path)   # Only read from .h5 file

    scr12 = Scores()
    scr12 = scr12.read_evl(sre12_evlfile)

    # filter the key so that it contains only trials for which we have scores.
    key12 = key12.filter(scr12.modelset, scr12.segset, keep=True)

    prior1 = effective_prior(0.01, 1, 1)  # P_tgt_A1 in SRE12
    prior2 = effective_prior(0.001, 1, 1)  # P_tgt_A2 in SRE12

    res12 = Results()
    res12.tar, res12.non, res12.knnon, res12.uknnon = scr12.get_tar_non(key12)

    res12.Pmiss, res12.Pfa = rocch(res12.tar, res12.non)
    res12.eer = rocch2eer(res12.Pmiss, res12.Pfa)

    minNormDcf12 = get_norm_min_dcf(res12, prior1, prior2)
    actNormDcf12 = get_norm_act_dcf(res12, prior1, prior2)

    # Print results based on uncalibrated scores
    print('Before score calibration\n')
    print('EER=%.2f; minNormDcf=%.3f; actNormDcf=%.3f\n' % (res12.eer*100, minNormDcf12, actNormDcf12))

    eer = res12.eer * 100
    dcf12 = minNormDcf12

    return eer, dcf12, res12