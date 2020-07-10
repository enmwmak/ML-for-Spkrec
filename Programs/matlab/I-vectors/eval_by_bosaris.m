function [eer, mindcf12, actdcf12, calw, pavfunc] = ...
    eval_by_bosaris(sre12_evlfile, sre12_keyfile, fast_keyloading, calw, pavfunc)
% Use Bosaris toolkit for evaluation. 
% be
% Input: 
%   sre12_evlfile      - Evaluation file that match the segment name of key file
%   keyfile            - Optional key file (work for SRE12 key file)
%   fast_keyloading    - Optinoal flag for fast keyfile loading (load a pre-stored .mat key file)
%   calw               - Calibration weights in 1x2 array
% Output:
%   eer                - Equal error rate
%   dcf12              - Minimum DCF as defined in SRE12
%   res12              - Result12 object in Bosaris toolkit
% Example:
%   evl2evl('evl/fw60/gplda60_male_cc4_1024c.evl', 'ndx/male/core-core_8k_male_cc4.ndx','evl/fw60/sre12_gplda60_male_cc4_1024c.evl');
%   [eer,dcf12,res12] = eval_by_bosaris('evl/fw60/sre12_gplda60_male_cc4_1024c.evl','../../key/NIST_SRE12_core_trial_key.v1',true);

% Setup environment for Bosaris
%bosaris_base = '~/so/Matlab/bosaris';
%addpath(genpath(bosaris_base));

% Note that user need to run scripts/re_arrange_evl.pl or evl2evl.m to produce an .evl file that can match the
% segment name of the key file.


% Set to true if .mat format of the key file has been saved from previous run
if ~exist('sre12_keyfile','var'),
    sre12_keyfile = '../../key/NIST_SRE12_core_trial_key.v1';          
end
if ~exist('fast_keyloading','var'),
    fast_keyloading = 'true';
end

% Setup Bosaris toolkit
%run('~/so/Matlab/bosaris/setup_bosaris.m');

% Alternatively, you may call the following private function
setup_bosaris('./bosaris_toolkit');       % Assuming ./bosaris_tk contains the toolkit

% Load key. If .mat file for key has been saved, it is quicker to load the .mat file
if (fast_keyloading == true),
    fprintf('Loading SRE12 key file %s\n',strcat(sre12_keyfile,'.mat'));
    key12 = Key.read(strcat(sre12_keyfile,'.mat')); 
else    
    fprintf('Loading SRE12 key file %s\n',strcat(sre12_keyfile,'.csv'));
    key12 = Key.read(strcat(sre12_keyfile,'.csv'));
    % Save SRE12 key to .mat file for improving loading speed
    fprintf('Saving SRE12 key as %s\n',strcat(sre12_keyfile,'.mat'));
    save_mat(key12,strcat(sre12_keyfile,'.mat'));
end

% Load SRE12 .evl file
fprintf('Loading SRE12 evl file %s\n',sre12_evlfile);
scr12 = Scores.read(sre12_evlfile);

% filter the key so that it contains only trials for which we have scores.
key12 = key12.filter(scr12.modelset,scr12.segset,true);

% Get normalized min DCF
res12 = Results12(scr12,key12);
prior1 = effective_prior(0.01,1,1);      % P_tgt_A1 in SRE12
prior2 = effective_prior(0.001,1,1);     % P_tgt_A2 in SRE12
minNormDcf12 = res12.get_norm_min_dcf(prior1,prior2);
actNormDcf12 = res12.get_norm_act_dcf(prior1,prior2);
Cllr = cllr(res12.tar,res12.non);
minCllr = min_cllr(res12.tar,res12.non);

% Print results based on uncalibrated scores
fprintf('Before score calibration\n');
fprintf('EER=%.2f; minNormDcf=%.3f; actNormDcf=%.3f; minCllr=%.3f; Cllr=%.3f\n',...
        res12.eer*100,minNormDcf12,actNormDcf12,minCllr,Cllr);

% Compute linear calibration weights and PAV function if they are not provided.    
if ~exist('calw','var'),
    [~,~,calw] = train_linear_calibration(res12.tar,res12.non,prior1,[],100,true);
end
if ~exist('pavfunc','var'),
    pavfunc = pav_calibration(res12.tar,res12.non,0.001);
end

% Perform linear score calibration
scr12_c = scr12;
scr12_c.scoremat = calw(1)*scr12.scoremat + calw(2);  % calw = [scale offset]
res12_c = Results12(scr12_c,key12);
minNormDcf12 = res12_c.get_norm_min_dcf(prior1,prior2);
actNormDcf12 = res12_c.get_norm_act_dcf(prior1,prior2);
Cllr = cllr(res12_c.tar,res12_c.non);
minCllr = min_cllr(res12_c.tar,res12_c.non);

% Print results based on calibrated scores
fprintf('After linear score calibration\n');
fprintf('EER=%.2f; minNormDcf=%.3f; actNormDcf=%.3f; minCllr=%.3f; Cllr=%.3f\n',...
        res12.eer*100,minNormDcf12,actNormDcf12,minCllr,Cllr);

% Perform PAV score calibration
scr12_c = scr12;
s = pavfunc(scr12.scoremat(:));
scr12_c.scoremat = reshape(s, size(scr12.scoremat));
res12_c = Results12(scr12_c,key12);
minNormDcf12 = res12_c.get_norm_min_dcf(prior1,prior2);
actNormDcf12 = res12_c.get_norm_act_dcf(prior1,prior2);
Cllr = cllr(res12_c.tar,res12_c.non);
minCllr = min_cllr(res12_c.tar,res12_c.non);

% Print results based on calibrated scores
fprintf('After PAV score calibration\n');
fprintf('EER=%.2f; minNormDcf=%.3f; actNormDcf=%.3f; minCllr=%.3f; Cllr=%.3f\n',...
        res12.eer*100,minNormDcf12,actNormDcf12,minCllr,Cllr);

eer = res12.eer*100;
mindcf12 = minNormDcf12;
actdcf12 = actNormDcf12;

return;
%%

function setup_bosaris(bosaris_base)
% Setup environment for Bosaris
addpath(genpath(bosaris_base));


%--------- FOR CALIBRATION OF THRESHOLD ONLY -------------------

% % Perform calibration. calw is the calibration weights [scale, offset]
% [~, calw] = linear_calibrate_scores(scr12,key12,[],100,prior1,true);
% scr12_c = scr12;
% scr12_c.scoremat = calw(1)*scr12.scoremat + calw(2);  % calw = [scale offset]
% res12_c = Results12(scr12_c,key12);
% minNormDcf12 = res12_c.get_norm_min_dcf(prior1,prior2);
% actNormDcf12 = res12_c.get_norm_act_dcf(prior1,prior2);
% 
% % Print results based on calibrated scores
% fprintf('After score calibration\n');
% fprintf('EER=%.2f; minNormDcf=%.3f; actNormDcf=%.3f\n',res12_c.eer*100,minNormDcf12,actNormDcf12);
% 
% % Compute compounded LLR
% load('mat/tgt-tst-scoremat.mat');       % Load scoremat (including trials not specified in SRE protocol)
% scr12_c = scr12;
% scr12_c.scoremat = calw(1)*scoremat + calw(2);   % Calibrate scores
% N = length(scr12_c.modelset);
% %scr12_cllr = llrTrans_simple2compound(scr12_c.scoremat,[ones(1,N)/N,1]/2);
% scr12_cllr = scr12_c.compoundLLR([ones(1,N)/N,1]/2);
% res12_cllr = Results12(scr12_cllr,key12);
% minNormDcf_cllr = res12_cllr.get_norm_min_dcf(prior1,prior2);
% actNormDcf_cllr = res12_cllr.get_norm_act_dcf(prior1,prior2);
% 
% % Print results based on calibrated compound LLR scores
% fprintf('Performance of compound LLR\n');
% fprintf('EER=%.2f; minNormDcf=%.3f; actNormDcf=%.3f\n',...
%     res12_cllr.eer*100,minNormDcf_cllr,actNormDcf_cllr);
% 
% % Use linear_calibrate_scores_dev_eval.m to perform score calibration
% 
% break
% 
% %%
% 
% sre10_keyfile = '../../../nist10/key/coreext-coreext.trialkey.mat';
% sre10_evlfile = '../../evl/fw60/sre10_gplda60_male_xphonecall-tel_1024c.evl';
% sre12_ndxfile = '../../ndx/male/core-core_8k_female_cc4.ndx';
% 
% % Convert SRE10 text key file to .mat file
% %key10 = Key.read('../../../nist10/key/coreext-coreext.trialkey.csv');
% %save_mat(key10,'../../../nist10/key/coreext-coreext.trialkey.mat');
% 
% fprintf('Loading SRE10 key file %s\n',sre10_keyfile);
% key10 = Key.read(sre10_keyfile);
% 
% %====================================================================
% % Use SRE10 key and scores for training the calibration parameters
% % Note that the scores are based on the PLDA model and T matrix trained 
% % for SRE12 so that the scores and i-vec are competable with those in SRE12
% %====================================================================
% 
% % Compute suff stats for targets and test segments
% comp_suf_stats('ubm/fw60/bg.mix.male.cmsY.1024c','../../../nist10/matlab/jfa/lst/fw60/male_xtarget_stats.lst','../../../nist10/matlab/jfa/stats/fw60/male_xtarget_mix_stats_1024c.mat',4);
% comp_suf_stats('ubm/fw60/bg.mix.male.cmsY.1024c','../../../nist10/matlab/jfa/lst/fw60/male_xtest_phonecall_tel_stats.lst','../../../nist10/matlab/jfa/stats/fw60/male_xtest-phonecall-tel_mix_stats_1024c.mat',4);
% 
% % Compute i-vectors for targets and test segments 
% comp_ivecs('ubm/fw60/bg.mix.male.cmsY.1024c','mat/fw60/male_mix_t500_1024c.mat','../../../nist10/matlab/jfa/stats/fw60/male_xtarget_mix_stats_1024c.mat','../../../nist10/matlab/jfa/mat/fw60/male_xtarget_mix_t500_w_1024c.mat',4);
% comp_ivecs('ubm/fw60/bg.mix.male.cmsY.1024c','mat/fw60/male_mix_t500_1024c.mat','../../../nist10/matlab/jfa/stats/fw60/male_xtest-phonecall-tel_mix_stats_1024c.mat','../../../nist10/matlab/jfa/mat/fw60/male_xtest-phonecall-tel_mix_t500_w_1024c.mat',4);
% 
% % Compute the scores of SRE10 and save it to .evl file (in SRE12 format)
% score_gplda_w({'../../../nist10/matlab/jfa/mat/fw60/male_xtarget_mix_t500_w_1024c.mat','../../../nist10/matlab/jfa/mat/fw60/male_xtarget_mix_t500_w_1024c.mat'},'','../../../nist10/matlab/jfa/mat/fw60/male_xtest-phonecall-tel_mix_t500_w_1024c.mat','../../../nist10/matlab/jfa/lst/fw60/male_xndx_phonecall_tel_stats.lst','','','None',sre10_evlfile,'mat/fw60/male_mix_t500_gplda_1024c.mat');
% 
% % Load SRE10 .evl file phonecall-tel
% fprintf('Loading SRE12 evl file %s\n',sre10_evlfile);
% scr10 = Scores.read(sre10_evlfile);
% 
% % Load SRE12 .ndx file for calibration
% fprintf('Loading SRE12 ndx file %s\n',sre10_ndxfile);
% ndx12 = Ndx.read(sre12_ndxfile);
% 
% % Use SRE10 to train a score calibrator and then calibrate the scores of SRE12
% [~,~,w_lin] = linear_calibrate_scores_dev_eval(scr10,scr12,key10,ndx12,[],10,prior1,true);

    
