import platform
import sys
from scipy.io import loadmat, savemat
import numpy as np
# import cupy as np
from numlines_ import numlines
from parse_list_ import parse_list
from mPLDA_fastGroupScoring1_ import mPLDA_fastGroupScoring1
from mPLDA_GroupScoring1_ import mPLDA_GroupScoring1
from mPLDA_GroupScoring5_ import mPLDA_GroupScoring5
from mPLDA_GroupScoring2_ import mPLDA_GroupScoring2
from mPLDA_train5_ import SImPLDA
from mPLDA_train4_ import SDmPLDA
import re
from utili_ import h52dict

def read_snr_file(snr_file):
    try:
        snr_file_fh= open(snr_file)
        snr = np.array(snr_file_fh.read().split()).reshape(-1, 1).astype(np.float64)
    finally:
        if snr_file_fh:
            snr_file_fh.close()
    return snr

def load_w_file(w_file):
    # i - vectors, spk_logical, spk_physical, num_frames
    temp = loadmat(w_file)
    return {'w': temp['w'], 'spk_logical': temp['spk_logical'], 'spk_physical': temp['spk_physical'], 'num_frames': temp['num_frames']}

def normalization(GPLDAModel, gplda_scr, tst_w, testutt, normType, normp, tgt_num):
    """
    % Create a hash table containing arrays of doubles [mu sigma] that have
    % been found in previous test sessions.
    % Use the session name as the key. <session_name,[mu sigma]>global sessionhash;
    """
    # preserve
    if normType == 'Znorm':
        scr = (gplda_scr - normp.znm.mu(tgt_num)) / normp.znm.sigma(tgt_num)


def snr3_score_gplda_w(target_w_file, tnorm_w_file, test_w_file, ndx_lstfile,
                                znorm_para_file, ztnorm_para_file, normType,
                                evlfile, GPLDA_file, tst_snr_file,  tgt_snr_file, opt):
    """
     function scores = snr3_score_gplda_w(target_w_file,tnorm_w_file,test_w_file,ndx_lstfile,...
                                     znorm_para_file, ztnorm_para_file, normType,...
                                     evlfile, GPLDA_file, tst_snr_file,  tgt_snr_file, opt)

     Perform mixture-PLDA scoring using SNR-dependent mixture of PLDA models and target i-vectors
     This function produces the results in my Interspeech14 paper.
     For compatibility with score_gplda_w_.py, the first 10 parameters are the same
     as those in score_gplda_w_.py
     Note 1: If target_w_file contains more than one file, the number of i-vectors in each file should
             be the same and their spk_logical should be identical.
     Note 2: The number of files in target_w_file and tgt_snr_file should be the same and their indexes
             should be aligned, i.e., target_w_file{i}.w{k} should align with tgt_snr_file{i}(k)

     Input:
       target_w_file        - Cell array containing SNR-dependent i-vectors files whose
       tnorm_w_file         - Remain here for backward compatibility. Should be empty
       test_w_file          - File containing test i-vectors
       ndx_lstfile          - List file specifying the evaluation trials
       znorm_para_file      - Remain here for backward compatibility. Should be empty
       ztnorm_para_file     - Remain here for backward compatibility. Should be empty
       normType             - Remain here for backward compatibility. Should be 'None'
       evlfile              - Output file in NIST SRE format
       GPLDA_file           - Cell array containing the Gaussian PLDA model files corresponding to the target files in target_w_file{}
       tst_snr_file         - A text file containing the SNR of test utterances in one column
       tgt_snr_file         - Cell array containing the SNR of target speakers' utts.
       opt                  - Optional parameters controlling the behaviour of the scoring process
     Author: M.W. Mak
     Date: Feb. 2014
     Example:
       cc4:opt.mode='scravg';snr3_score_gplda_w({'mat/fw60/male_target-tel_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-15dB_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-06dB_mix_t500_w_1024c.mat'},'','mat/fw60/male_test-tel-phn_mix_t500_w_1024c.mat','lst/fw60/male_ndx_cc4_stats.lst','','','None','evl/fw60/gplda60_male_cc4_1024c.evl','mat/fw60/male_mix_t500_mgplda_cln-15dB-06dB_1024c.mat','../../snr/male/male_test-tel-phn_stats.snr',{'../../snr/male/male_target-tel_stats.snr','../../snr/male/male_target-tel-15dB_stats.snr','../../snr/male/male_target-tel-06dB_stats.snr'},opt); cd ../../; system('scripts/re_arrange_evl.pl -ndx ndx/male/core-core_8k_male_cc4.ndx -evl matlab/jfa/evl/fw60/gplda60_male_cc4_1024c.evl | more > evl/fw60/sre12_gplda60_male_cc4_1024c.evl; scripts/comperr.pl -key /corpus/nist12/doc/NIST_SRE12_core_trial_key.v1.csv -evl evl/fw60/sre12_gplda60_male_cc4_1024c.evl -sscr scr/fw60/sre12_gplda60_male_cc4_1024c-spk.scr -iscr scr/fw60/sre12_gplda60_male_cc4_1024c-imp.scr -ikscr scr/fw60/sre12_gplda60_male_cc4_1024c-knownimp.scr -iukscr scr/fw60/sre12_gplda60_male_cc4_1024c-unknownimp.scr -ad nist12 -vad 1 -cc 4'); cd matlab/jfa;
       cc5:opt.mode='scravg';snr3_score_gplda_w({'mat/fw60/male_target-tel_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-15dB_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-06dB_mix_t500_w_1024c.mat'},'','mat/fw60/male_test-tel-phn_mix_t500_w_1024c.mat','lst/fw60/male_ndx_cc5_stats.lst','','','None','evl/fw60/gplda60_male_cc5_1024c.evl','mat/fw60/male_mix_t500_mgplda_cln-15dB-06dB_1024c.mat','../../snr/male/male_test-tel-phn_stats.snr',{'../../snr/male/male_target-tel_stats.snr','../../snr/male/male_target-tel-15dB_stats.snr','../../snr/male/male_target-tel-06dB_stats.snr'},opt); cd ../../; system('scripts/re_arrange_evl.pl -ndx ndx/male/core-core_8k_male_cc5.ndx -evl matlab/jfa/evl/fw60/gplda60_male_cc5_1024c.evl | more > evl/fw60/sre12_gplda60_male_cc5_1024c.evl; scripts/comperr.pl -key /corpus/nist12/doc/NIST_SRE12_core_trial_key.v1.csv -evl evl/fw60/sre12_gplda60_male_cc5_1024c.evl -sscr scr/fw60/sre12_gplda60_male_cc5_1024c-spk.scr -iscr scr/fw60/sre12_gplda60_male_cc5_1024c-imp.scr -ikscr scr/fw60/sre12_gplda60_male_cc5_1024c-knownimp.scr -iukscr scr/fw60/sre12_gplda60_male_cc5_1024c-unknownimp.scr -ad nist12 -vad 1 -cc 5'); cd matlab/jfa;

    """

    # % opt.mode = 'scravg': Scoring test ivec against target ivecs individually
    # % opt.mode = 'ivcavg'   : Scoring test ivec against the averaged i-vecs within each SNR group without averaging the SNR
    # % opt.mode = 'ivcsnravg': Scoring test ivec against the averaged i-vecs within each SNR group using averaged SNR
    # % opt.mode = 'fastscr'  : Scoring test ivec against target ivecs individually, using fast scoring method
    # % opt.mtype = 'SImPLDA' : SNR-independent mixture of PLDA (Hinton's model, use i-vecs for alignment)
    # % opt.mtype = 'SDmPLDA' : SNR-dependent mixture of PLDA (use SNR for alignment)
    try:
        opt.mode
    except AttributeError:
        opt.mode = 'scravg'
    try:
        opt.mtype
    except AttributeError:
        opt.mtype = 'SDmPLDA'

    # Set up path to use GPLDA package
    if platform.system() == 'Windows':
        sys.path.append('D:/so/Matlab/PLDA/BayesPLDA')
    else:
        sys.path.append('~/so/Matlab/PLDA/BayesPLDA')

    # Load SNR of tst utterances
    print('Loading ', tst_snr_file)
    tst_snr = read_snr_file(tst_snr_file)

    # # Load SNR of tgt utterances
    tgt_snr = np.empty(shape=(len(tgt_snr_file), 1), dtype=object)
    for i in range(len(tgt_snr_file)):
        print('Loading ', tgt_snr_file[i])
        tgt_snr[i, 0] = read_snr_file(tgt_snr_file[i])

    # Load mGPLDA model structure to obtain the structure GPLDAModel
    print('Loading ', GPLDA_file)
    if opt.mtype == 'SImPLDA':
        model = SImPLDA()
    elif opt.mtype == 'SDmPLDA':
        model = SDmPLDA()

    GPLDAModel = model
    GPLDAModel.load_model(GPLDA_file)

    # Load SNR-dependent i-vectors of target speakers.
    # Concatenate the i-vectors, spk_logical, spk_physical, num_frames in the input target_w_files
    tgt = np.empty(shape=(len(target_w_file), len(target_w_file)), dtype=object)

    for i in range(len(target_w_file)):
        tgt[i, 0] = load_w_file(target_w_file[i])

    # Load the test i-vectors
    print('Loading ', test_w_file)
    tst = load_w_file(test_w_file)

    # Load NORM models (w)
    if normType == 'Znorm':
        print('Loading ', znorm_para_file)
        # preseve
    elif normType == 'Tnorm':
        print('Loading ', znorm_para_file)
        # preseve
    elif normType == 'ZTnorm1':
        print('Loading ', znorm_para_file)
        # preseve
    elif normType == 'ZTnorm2':
        print('Loading ', znorm_para_file)
        # preseve
    elif normType == 'None':
        print('No norm')
        normp = {'tnm':[],'znm':[],'tnm_w':[],'ztnm':[]}
    else:
        print('Incorrect norm type')

    print('Scoring mode: %s\n' % opt.mode)
    num_tests = numlines(ndx_lstfile)
    scores = np.zeros(shape=(num_tests,1), dtype=np.float64)
    ndx = {'spk_logical': parse_list(ndx_lstfile)}
    n_tstutt = len(tst['spk_logical'])
    C_target = np.empty(shape=(num_tests, 1), dtype=object)
    C_testutt = np.empty(shape=(num_tests, 1), dtype=object)
    C_channel = np.empty(shape=(num_tests, 1), dtype=object)

    # Start scoring test i-vec against target i-vec based on the target-test pairing in .ndx file
    for i in range(num_tests):
        session_name = ndx['spk_logical'][i]
        field = session_name.split(':')
        target = field[0]
        testutt = field[1]
        field = testutt.split('_')
        channel = field[-1].lower()

        # Find the index k of the test utt
        k = np.where((testutt == tst['spk_logical']) == True)[0][0]
        tst_w = tst['w'][k, :].reshape(1, -1)        # k should be a scalar

        # Find target session of the current target speaker
        tgt_sessions = np.where((target == tgt[0, 0]['spk_logical']) == True)[0]

        # Exclude short target utterances
        n_tgt_frms = tgt[0, 0]['num_frames'][tgt_sessions]
        tgt_ss = tgt_sessions[np.where(n_tgt_frms >= 1000)[0]]

        if tgt_ss.size != 0:
            tgt_sessions = tgt_ss

        # Prepare the target-ivecs for mPLDA scoring
        n_tgt_sess = len(tgt_sessions)

        # Make sure that target sessions exist
        assert n_tgt_sess > 0, print('%d: Missing sessions of %s in snr3_score_gplda_w_.py' % (i, target))

        # Perform different types of scoring based on the mode para in opt
        if opt.mode == 'scravg' or opt.mode == 'fastscr':
            # Score test i-vec with target i-vec individually (default).
            # Pack the i-vectors of target's training sessions into one matrix
            tgt_w = np.zeros(shape=(n_tgt_sess*len(tgt), tst['w'].shape[1]), dtype=np.float64)
            tgt_sess_snr = np.zeros(shape=(n_tgt_sess * len(tgt), 1), dtype=np.float64)
            for s in range(len(tgt)):
                tgt_w[s * n_tgt_sess: (s+1) * n_tgt_sess, :] = tgt[s, 0]['w'][tgt_sessions, :]
                tgt_sess_snr[s * n_tgt_sess: (s+1) * n_tgt_sess] = tgt_snr[s, 0][tgt_sessions]
            # Compute the scores of current tst ivec against all target's i-vecs
            #  Uncomment the following line for estimating the scoring time using profiler
            #  tgt_w = mean(tgt_w,1); tgt_sess_snr = mean(tgt_sess_snr);
            if opt.mode == 'fastscr':
                gplda_scr = mPLDA_fastGroupScoring1(GPLDAModel, tgt_w.T, tst_w.T, tgt_sess_snr, tst_snr[k][0])
            elif opt.mtype == 'SDmPLDA':
                gplda_scr = mPLDA_GroupScoring1(GPLDAModel, tgt_w.T, tst_w.T, tgt_sess_snr, tst_snr[k][0])
            else:
                gplda_scr = mPLDA_GroupScoring5(GPLDAModel, tgt_w.T, tst_w.T)

        elif opt.mode == 'ivcavg':
            # Score test i-vec with the average of SNR-dependent target i-vec without averaging the SNR
            tgt_w = np.zeros(shape=(len(tgt), tst['w'].shape[1]), dtype=np.float64)
            tgt_sess_snr = np.zeros(shape=(n_tgt_sess*len(tgt), 1), dtype=np.float64)
            for s in range(len(tgt)):
                tgt_w[s, :] = np.mean(tgt[s, 0]['w'][tgt_sessions], axis=0).reshape(1, -1)
                tgt_sess_snr[s * n_tgt_sess: (s + 1) * n_tgt_sess] = tgt_snr[s, 0][tgt_sessions]
            # Compute the scores of current tst ivec against the averaged target i-vec in each SNR group
            gplda_scr = mPLDA_GroupScoring2(GPLDAModel, tgt_w.T, tst_w.T, tgt_sess_snr, tst_snr[k][0])

        elif opt.mode == 'ivcsnravg':
            # Score test i-vec with the average of SNR-dependent target i-vec using averaged SNR
            tgt_w = np.zeros(shape=(len(tgt), tst['w'].shape[1]), dtype=np.float64)
            tgt_sess_snr = np.zeros(shape=(len(tgt), 1), dtype=np.float64)
            for s in range(len(tgt)):
                tgt_w[s, :] = np.mean(tgt[s, 0]['w'][tgt_sessions], axis=0).reshape(1, -1)
                tgt_sess_snr[s] = np.mean(tgt_snr[s, 0][tgt_sessions])
            gplda_scr = mPLDA_GroupScoring1(GPLDAModel, tgt_w.T, tst_w.T, tgt_sess_snr, tst_snr[k][0])

        # Perform score normalization (if necessary) and compute the mean PLDA score
        if normType == 'None':
            scores[i] = np.mean(gplda_scr)
        else:
            tgt_num = np.where((target == normp['znm']['spk_id']) == True)   # Find the target number (index in normp)
            scores[i] = np.mean(normalization(GPLDAModel, gplda_scr, tst_w, testutt, normType, normp, tgt_num))

        if np.mod(i, 10000) == 0:
            print('(%d/%d) %s,%s: %f\n' % (i, num_tests, target, testutt, scores[i]))

        # Store the score, target and tst session in cell arrays for saving later
        C_target[i, 0] = target
        C_testutt[i, 0] = testutt
        C_channel[i, 0] = channel

    # Compute the compound LLR
    # c_scores = compound_llr(scores, tstscore, C_testutt, tst);
    c_scores = scores

    # Save the score to .evl file
    try:
        fp = open(evlfile, 'w')
        for i in range(num_tests):  # num_tests
            testsegfile = re.split('_A|_B', C_testutt[i, 0])[0] + '.sph'
            fp.write('%s,%s,%c,%.7f\n' % (C_target[i, 0], testsegfile, C_channel[i, 0], c_scores[i]))
    finally:
        if fp:
            fp.close()

