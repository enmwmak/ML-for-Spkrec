% Test PLDA model using clean (original) utterances
% Use Training Data Set I and Set II to train PLDA.
% This script produces the results in Table 3 of the T-ASLP paper.

% Use Dataset 1 or Dataset 2 as defined in T-ASLP paper to train PLDA (male)
% This script assumes that the following directories already exist, and it will
% write to these directories
%   ./mat/fw60
%   ./evl/fw60
%   ./bosaris
%
% The default setting of this script will produce the results in CC4 (male, Set I)
% of Table III of the T-ASLP paper. Note that because the PLDA model is initialized
% randomly (see PLDA_train5.m), the results will be slightly different from run-to-run.
%
% Author: Man-Wai MAK, Dept. of EIE, The Hong Kong Polytechnic University
% Version: 1.0
% Date: Aug 2015
%
% This file is subject to the terms and conditions defined in
% file 'license.txt', which is part of this source code package.

clear; close all;

% Define constants
dataset = 1;
mtype = 'PLDA';                             % Can only be 'PLDA'
opt.mode = 'scravg';                        % Use score averaging
opt.mtype = mtype;
K = 1;                                      % Must be 1 for PLDA model

% Load training i-vectors and SNR information
trndatafile = sprintf('mat/fw60/male_target-dataset%d_mix_t500_w_1024c.mat',dataset);
fprintf('Loading datafile %s\n',trndatafile);
tgt = load(trndatafile);

% Define PLDA and evaluation files and train PLDA model
pldafile = sprintf('mat/fw60/male_mix_t500_%s_dataset%d-K%d_1024c.mat',mtype,dataset,K);    
fprintf('Training %s\n',pldafile);
GPLDAModel = comp_GPLDA1(tgt.w, tgt.spk_logical, pldafile, 150);

ccres = cell(2);     
cc = [4];

% Define target i-vector file for scoring
tgt_ivec_file = {'mat/fw60/male_target-tel_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-15dB_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-06dB_mix_t500_w_1024c.mat'};

% Define test i-vector file for scoring
test_ivec_file = 'mat/fw60/male_test-tel-phn_mix_t500_w_1024c.mat';

for j = 1:length(cc),
    evlfile = sprintf('evl/fw60/male_mix_t500_%s_dataset%d-K%d_1024c_cc%d.evl',mtype,dataset,K,cc(j));
    sre12_evlfile = sprintf('evl/fw60/sre12_male_mix_t500_%s_dataset%d-K%d_1024c_cc%d.evl',mtype,dataset,K,cc(j));
    res12_file = sprintf('bosaris/male_cleantest_%s_dataset%d-K%d_cc%d.mat',mtype,dataset,K,cc(j));
    ndx_lstfile = sprintf('lst/fw60/male_ndx_cc%d_stats.lst',cc(j));
    ndxfile = sprintf('ndx/male/core-core_8k_male_cc%d.ndx',cc(j));

    score_gplda_w(tgt_ivec_file,'',test_ivec_file,ndx_lstfile,'','','None',evlfile,pldafile); 
    evl2evl(evlfile, ndxfile,sre12_evlfile);
    [ccres{j}.eer, ccres{j}.dcf12, ccres{j}.res12] = eval_by_bosaris(sre12_evlfile);

    res12 = ccres{j}.res12;
    eer = ccres{j}.eer;
    dcf12 = ccres{j}.dcf12;
    fprintf('Saving result to %s\n',res12_file);
    save(res12_file,'res12','eer','dcf12');
end

break;

%% Ignre code below this line

