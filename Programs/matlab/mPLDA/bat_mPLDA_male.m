% Use Dataset 1 or Dataset 2 as defined in T-ASLP paper to train SImPLDA
% and SDmPLDA models (male)
% This script assumes that the following directories already exist, and it will
% write to these directories
%   ./mat/fw60
%   ./evl/fw60
%   ./bosaris
%
% The default setting of this script will produce the results in CC4 (male, Set I)
% of Table III of the T-ASLP paper. Note that because the mPLDA model is initialized
% randomly (see mPLDA_train5.m), the results will be slightly different from run-to-run.
%
% Author: Man-Wai MAK, Dept. of EIE, The Hong Kong Polytechnic University
% Version: 1.0
% Date: Aug 2015
%
% This file is subject to the terms and conditions defined in
% file 'license.txt', which is part of this source code package.
 
clear; close all;

% Define constants
dataset = 1;                              % Can be 1 or 2 (Set I and Set II in paper)
mtype = 'SDmPLDA';                        % Can be 'SImPLDA' or 'SDmPLDA'  
opt.mode = 'scravg';                      % Use score averaging
opt.mtype = mtype;
nMixSet = 3;                              % No. of mixtures, e.g., nMixSet = [2 3 4]
cc = 4;                                   % SRE12 common conditions, e.g., cc = [4 5]

% Determine which training algorithm to run
switch mtype
    case 'SImPLDA'
        mPLDA_trainfunc = @mPLDA_train5;          % Use SNR-independent mPLDA
    case 'SDmPLDA'
        mPLDA_trainfunc = @mPLDA_train4;          % Use SNR-dependent mPLDA   
    otherwise
        mPLDA_trainfunc = @mPLDA_train3;          % Use old version of SNR-dependent mPLDA   
end

% Load training i-vectors and SNR information
trndatafile = sprintf('mat/fw60/male_target-dataset%d_mix_t500_w_1024c.mat',dataset);
fprintf('Loading datafile %s\n',trndatafile);
tgt = load(trndatafile);

% Define target i-vector file and target SNR file for scoring
tgt_ivec_file = {'mat/fw60/male_target-tel_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-15dB_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-06dB_mix_t500_w_1024c.mat'};
tgt_snr_file = {'snr/male/male_target-tel_stats.snr','snr/male/male_target-tel-15dB_stats.snr','snr/male/male_target-tel-06dB_stats.snr'};

% Define test i-vec file and test snr file for scoring
tst_ivec_file = 'mat/fw60/male_test-tel-phn_mix_t500_w_1024c.mat';
tst_snr_file = 'snr/male/male_test-tel-phn_stats.snr';

% For each number of mixtures k, train and score an mPLDA model
for r = 1:length(nMixSet),
    k = nMixSet(r);
    
    % Define mPLDA and evaluation files
    mpldafile = sprintf('mat/fw60/male_mix_t500_%s_dataset%d-K%d_1024c.mat',mtype,dataset,k);    
    fprintf('Training %s\n',mpldafile);
    GPLDAModel = comp_mGPLDA(tgt.w, tgt.snr, tgt.spk_logical, mpldafile, 150, k, mPLDA_trainfunc);

    for j = 1:length(cc),
        evlfile = sprintf('evl/fw60/male_mix_t500_%s_dataset%d-K%d_1024c_cc%d.evl',mtype,dataset,k,cc(j));
        fprintf('evlfile=%s\n',evlfile);
        sre12_evlfile = sprintf('evl/fw60/sre12_male_mix_t500_%s_dataset%d-K%d_1024c_cc%d.evl',mtype,dataset,k,cc(j));
        res12_file = sprintf('bosaris/male_cleantest_%s_dataset%d-K%d_cc%d.mat',mtype,dataset,k,cc(j));
        ndx_lstfile = sprintf('lst/fw60/male_ndx_cc%d_stats.lst',cc(j));
        ndxfile = sprintf('ndx/male/core-core_8k_male_cc%d.ndx',cc(j));

        snr3_score_gplda_w(tgt_ivec_file,'',tst_ivec_file,ndx_lstfile,'','','None',evlfile,...
                            mpldafile,tst_snr_file,tgt_snr_file,opt);    	
        evl2evl(evlfile, ndxfile, sre12_evlfile);
        [ccres{k,j}.eer, ccres{k,j}.dcf12, ccres{k,j}.res12] = eval_by_bosaris(sre12_evlfile);

        res12 = ccres{k,j}.res12;
        eer = ccres{k,j}.eer;
        dcf12 = ccres{k,j}.dcf12;
        fprintf('Saving result to %s\n',res12_file);
        save(res12_file,'res12','eer','dcf12');            
        
    end
end
break;
