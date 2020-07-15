% function scores = snr3_score_gplda_w(target_w_file,tnorm_w_file,test_w_file,ndx_lstfile,...
%                                 znorm_para_file, ztnorm_para_file, normType,...
%                                 evlfile, GPLDA_file, tst_snr_file,  tgt_snr_file, opt)
%
% Perform mixture-PLDA scoring using SNR-dependent mixture of PLDA models and target i-vectors
% This function produces the results in my Interspeech14 paper.
% For compatibility with score_gplda_w.m, the first 10 parameters are the same
% as those in score_gplda_w.m
% Note 1: If target_w_file contains more than one file, the number of i-vectors in each file should
%         be the same and their spk_logical should be identical.
% Note 2: The number of files in target_w_file and tgt_snr_file should be the same and their indexes
%         should be aligned, i.e., target_w_file{i}.w{k} should align with tgt_snr_file{i}(k)
%
% Input:
%   target_w_file        - Cell array containing SNR-dependent i-vectors files whose 
%   tnorm_w_file         - Remain here for backward compatibility. Should be empty
%   test_w_file          - File containing test i-vectors
%   ndx_lstfile          - List file specifying the evaluation trials
%   znorm_para_file      - Remain here for backward compatibility. Should be empty
%   ztnorm_para_file     - Remain here for backward compatibility. Should be empty
%   normType             - Remain here for backward compatibility. Should be 'None'
%   evlfile              - Output file in NIST SRE format
%   GPLDA_file           - Cell array containing the Gaussian PLDA model files corresponding to the target files in target_w_file{}
%   tst_snr_file         - A text file containing the SNR of test utterances in one column
%   tgt_snr_file         - Cell array containing the SNR of target speakers' utts.
%   opt                  - Optional parameters controlling the behaviour of the scoring process
% Author: M.W. Mak
% Date: Feb. 2014
% Example:
%   cc4:opt.mode='scravg';snr3_score_gplda_w({'mat/fw60/male_target-tel_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-15dB_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-06dB_mix_t500_w_1024c.mat'},'','mat/fw60/male_test-tel-phn_mix_t500_w_1024c.mat','lst/fw60/male_ndx_cc4_stats.lst','','','None','evl/fw60/gplda60_male_cc4_1024c.evl','mat/fw60/male_mix_t500_mgplda_cln-15dB-06dB_1024c.mat','../../snr/male/male_test-tel-phn_stats.snr',{'../../snr/male/male_target-tel_stats.snr','../../snr/male/male_target-tel-15dB_stats.snr','../../snr/male/male_target-tel-06dB_stats.snr'},opt); cd ../../; system('scripts/re_arrange_evl.pl -ndx ndx/male/core-core_8k_male_cc4.ndx -evl matlab/jfa/evl/fw60/gplda60_male_cc4_1024c.evl | more > evl/fw60/sre12_gplda60_male_cc4_1024c.evl; scripts/comperr.pl -key /corpus/nist12/doc/NIST_SRE12_core_trial_key.v1.csv -evl evl/fw60/sre12_gplda60_male_cc4_1024c.evl -sscr scr/fw60/sre12_gplda60_male_cc4_1024c-spk.scr -iscr scr/fw60/sre12_gplda60_male_cc4_1024c-imp.scr -ikscr scr/fw60/sre12_gplda60_male_cc4_1024c-knownimp.scr -iukscr scr/fw60/sre12_gplda60_male_cc4_1024c-unknownimp.scr -ad nist12 -vad 1 -cc 4'); cd matlab/jfa;
%   cc5:opt.mode='scravg';snr3_score_gplda_w({'mat/fw60/male_target-tel_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-15dB_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-06dB_mix_t500_w_1024c.mat'},'','mat/fw60/male_test-tel-phn_mix_t500_w_1024c.mat','lst/fw60/male_ndx_cc5_stats.lst','','','None','evl/fw60/gplda60_male_cc5_1024c.evl','mat/fw60/male_mix_t500_mgplda_cln-15dB-06dB_1024c.mat','../../snr/male/male_test-tel-phn_stats.snr',{'../../snr/male/male_target-tel_stats.snr','../../snr/male/male_target-tel-15dB_stats.snr','../../snr/male/male_target-tel-06dB_stats.snr'},opt); cd ../../; system('scripts/re_arrange_evl.pl -ndx ndx/male/core-core_8k_male_cc5.ndx -evl matlab/jfa/evl/fw60/gplda60_male_cc5_1024c.evl | more > evl/fw60/sre12_gplda60_male_cc5_1024c.evl; scripts/comperr.pl -key /corpus/nist12/doc/NIST_SRE12_core_trial_key.v1.csv -evl evl/fw60/sre12_gplda60_male_cc5_1024c.evl -sscr scr/fw60/sre12_gplda60_male_cc5_1024c-spk.scr -iscr scr/fw60/sre12_gplda60_male_cc5_1024c-imp.scr -ikscr scr/fw60/sre12_gplda60_male_cc5_1024c-knownimp.scr -iukscr scr/fw60/sre12_gplda60_male_cc5_1024c-unknownimp.scr -ad nist12 -vad 1 -cc 5'); cd matlab/jfa;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function scores = snr3_score_gplda_w(target_w_file,tnorm_w_file,test_w_file,ndx_lstfile,...
                                znorm_para_file, ztnorm_para_file, normType,...
                                evlfile, GPLDA_file, tst_snr_file,  tgt_snr_file, opt)
                            
% opt.mode = 'scravg'   : Scoring test ivec against target ivecs individually
% opt.mode = 'ivcavg'   : Scoring test ivec against the averaged i-vecs within each SNR group without averaging the SNR
% opt.mode = 'ivcsnravg': Scoring test ivec against the averaged i-vecs within each SNR group using averaged SNR
% opt.mode = 'fastscr'  : Scoring test ivec against target ivecs individually, using fast scoring method
% opt.mtype = 'SImPLDA' : SNR-independent mixture of PLDA (Hinton's model, use i-vecs for alignment)
% opt.mtype = 'SDmPLDA' : SNR-dependent mixture of PLDA (use SNR for alignment)
if ~any(strcmp('mode',fieldnames(opt))),
    opt.mode = 'scravg';
end 

if ~any(strcmp('mtype',fieldnames(opt))),
    opt.mtype = 'SDmPLDA';
end 


if nargin ~= 11 && nargin ~=12,
    disp('Need at 11 or 12 input arguments');
    return;
end

% Set up path to use GPLDA package
if (strcmp(computer,'PCWIN')==1 || strcmp(computer,'PCWIN64')==1),
    addpath 'D:/so/Matlab/PLDA/BayesPLDA';  
else
    addpath '~/so/Matlab/PLDA/BayesPLDA';
end    

% Load SNR of tst utterances
disp(['Loading ' tst_snr_file]);
tst_snr = load(tst_snr_file);

% Load SNR of tgt utterances
tgt_snr = cell(length(tgt_snr_file),1);
for i=1:length(tgt_snr_file),
    disp(['Loading ' tgt_snr_file{i}]);
    tgt_snr{i} = load(tgt_snr_file{i});
end

% Load mGPLDA model structure to obtain the structure GPLDAModel
disp(['Loading ' GPLDA_file]);
temp = load(GPLDA_file);
GPLDAModel = temp.GPLDAModel; clear temp;

% Load SNR-dependent i-vectors of target speakers.
% Concatenate the i-vectors, spk_logical, spk_physical, num_frames in the input target_w_files
tgt = cell(length(target_w_file)); 
for i=1:length(target_w_file),
    tgt{i} = struct('w',[],'spk_logical',[],'spk_physical',[],'num_frames',[]);
    disp(['Loading ' target_w_file{i}]);
    tgt{i} = load(target_w_file{i});       % Load w and spk_logical of target speakers
end
clear trn;

% Load the test i-vectors
disp(['Loading ' test_w_file]);
tst = load(test_w_file);

% Load NORM models (w)
switch (normType)
    case 'Znorm'
        disp(['Loading ' znorm_para_file]);
        normp.znm = load(znorm_para_file);
    case {'Tnorm'}
        disp(['Loading ' tnorm_w_file]);
        normp.tnm = load(tnorm_w_file);
        normp.tnm_w = normp.tnm.w;
    case {'ZTnorm1'}
        disp(['Loading ' znorm_para_file]);
        normp.znm = load(znorm_para_file);        
        disp(['Loading ' tnorm_w_file]);
        normp.tnm = load(tnorm_w_file);
        normp.tnm_w = normp.tnm.w;
    case {'ZTnorm2'}
        disp(['Loading ' znorm_para_file]);
        normp.znm = load(znorm_para_file);        
        disp(['Loading ' tnorm_w_file]);
        normp.tnm = load(tnorm_w_file);
        normp.tnm_w = normp.tnm.w;
        disp(['Loading ' ztnorm_para_file]);
        normp.ztnm = load(ztnorm_para_file);
    case {'None'}
        disp('No norm');
        normp = struct('tnm',[],'znm',[],'tnm_w',[],'ztnm',[]);
    otherwise
        disp('Incorrect norm type');
        return;
end

% Compute the score of all trials, calling PLDA_GroupScoring one trial at a time.
fprintf('Scoring mode: %s\n', opt.mode);
num_tests = numlines(ndx_lstfile);
scores = zeros(num_tests,1);
ndx.spk_logical = parse_list(ndx_lstfile);
n_tstutt = length(tst.spk_logical);
C_target = cell(n_tstutt,1);
C_testutt = cell(n_tstutt,1);
C_channel = cell(n_tstutt,1);


% Start scoring test i-vec against target i-vec based on the target-test pairing in .ndx file
tic;
for i=1:num_tests,                           % For each trial
    session_name = ndx.spk_logical{i};       % Test session name, eg., 100396:taetxk_sre12_B
    field = rsplit(':',session_name);        % Split spk_logical into target and test utt 
    target = field{1};                       % e.g., 100396
    testutt = field{2};                      % e.g., taetxk_sre12_B
    field = rsplit('_',testutt);
    channel = lower(field{end});             % e.g., b
        
    % Find the index k of the test utt
    k = find(strncmp(testutt, tst.spk_logical,length(testutt))==1); 
    tst_w = tst.w(k,:);                              % k should be a scalar 
   
    % Find target session of the current target speaker
    tgt_sessions = find(strncmp(target, tgt{1}.spk_logical,length(target))==1);  

    % Exclude short target utterances
    n_tgt_frms = tgt{1}.num_frames(tgt_sessions);
    tgt_ss = tgt_sessions(n_tgt_frms>=1000);
    if (isempty(tgt_ss)~=1)
        tgt_sessions = tgt_ss;
    end
    
    % Prepare the target-ivecs for mPLDA scoring
    n_tgt_sess = length(tgt_sessions);

    % Make sure that target sessions exist
    assert(n_tgt_sess>0,sprintf('%d: Missing sessions of %s in snr3_score_gplda_w.m',i,target));    
    
    % Perform different types of scoring based on the mode para in opt
    switch(opt.mode)
        case {'scravg','fastscr'}
            % Score test i-vec with target i-vec individually (default). 
            % Pack the i-vectors of target's training sessions into one matrix
            tgt_w = zeros(n_tgt_sess*length(tgt),size(tst.w,2));
            tgt_sess_snr = zeros(n_tgt_sess*length(tgt),1);
            for s=1:length(tgt),
                tgt_w((s-1)*n_tgt_sess+1 : s*n_tgt_sess, :) = tgt{s}.w(tgt_sessions,:);
                tgt_sess_snr((s-1)*n_tgt_sess+1 : s*n_tgt_sess) = tgt_snr{s}(tgt_sessions);
            end
            % Compute the scores of current tst ivec against all target's i-vecs
            % Uncomment the following line for estimating the scoring time using profiler
            % tgt_w = mean(tgt_w,1); tgt_sess_snr = mean(tgt_sess_snr); 
            if (strcmp(opt.mode,'fastscr')==1),
                gplda_scr = mPLDA_fastGroupScoring1(GPLDAModel,tgt_w',tst_w',tgt_sess_snr,tst_snr(k));
            elseif (strcmp(opt.mtype,'SDmPLDA')==1),
                gplda_scr = mPLDA_GroupScoring1(GPLDAModel,tgt_w',tst_w',tgt_sess_snr,tst_snr(k));
            else
                gplda_scr = mPLDA_GroupScoring5(GPLDAModel,tgt_w',tst_w');  % mPLDA model should be trained by mPLDA_train5.m
            end
        case {'ivcavg'}
            % Score test i-vec with the average of SNR-dependent target i-vec without averaging the SNR
            tgt_w = zeros(length(tgt),size(tst.w,2));
            tgt_sess_snr = zeros(n_tgt_sess*length(tgt),1);
            for s=1:length(tgt),
                tgt_w(s, :) = mean(tgt{s}.w(tgt_sessions,:),1);
                tgt_sess_snr((s-1)*n_tgt_sess+1 : s*n_tgt_sess) = tgt_snr{s}(tgt_sessions);
            end
            % Compute the scores of current tst ivec against the averaged target i-vec in each SNR group
            gplda_scr = mPLDA_GroupScoring2(GPLDAModel,tgt_w',tst_w',tgt_sess_snr,tst_snr(k));

        case {'ivcsnravg'}
            % Score test i-vec with the average of SNR-dependent target i-vec using averaged SNR
            tgt_w = zeros(length(tgt),size(tst.w,2));
            tgt_sess_snr = zeros(length(tgt),1);
            for s=1:length(tgt),
                tgt_w(s, :) = mean(tgt{s}.w(tgt_sessions,:),1);
                tgt_sess_snr(s) = mean(tgt_snr{s}(tgt_sessions));
            end
            % Compute the scores of current tst ivec against the averaged target i-vec in each SNR group using averaged SNR
            gplda_scr = mPLDA_GroupScoring1(GPLDAModel,tgt_w',tst_w',tgt_sess_snr,tst_snr(k));

    end
        
    
    % Perform score normalization (if necessary) and compute the mean PLDA score
    if (strcmp(normType,'None')==1),
        scores(i) = mean(gplda_scr);
    else
        tgt_num = find(strcmp(target, normp.znm.spk_id)==1);            % Find the target number (index in normp)
        scores(i) = mean(normalization(GPLDAModel, gplda_scr, tst_w, testutt, normType, normp, tgt_num));
    end
          
    if mod(i,10000)==0,
        fprintf('(%d/%d) %s,%s: %f\n',i,num_tests,target,testutt,scores(i));
    end
    
    % Store the score, target and tst session in cell arrays for saving later
    C_target{i} = target;
    C_testutt{i} = testutt;
    C_channel{i} = channel;       
end
toc;

% Compute the compound LLR
%c_scores = compound_llr(scores, tstscore, C_testutt, tst);
c_scores = scores;

% Save the score to .evl file
fp = fopen(evlfile,'w');
for i=1:num_tests,
    testsegfile = rsplit('_A|_B',C_testutt{i});
    testsegfile{1} = strcat(testsegfile{1},'.sph');
    fprintf(fp,'%s,%s,%c,%.7f\n', C_target{i}, testsegfile{1}, C_channel{i}, c_scores(i));    
end
fclose(fp);

disp(['Scores saved to ' evlfile]);

return;


%% private function

function c_scores = compound_llr(scores, tstscore, C_testutt, tst)
% This function implements Eq. 14 of the paper 
% "Knowing the non-target speakers: The effect of the i-vector population for
% PLDA training in speaker recognition.
num_tests = length(scores);
c_scores = zeros(num_tests,1);
for i=1:num_tests,
    k = find(strncmp(C_testutt{i},tst.spk_logical,length(C_testutt{i}))==1); % Find the index of the test utt
    kn_nontgt_scr = tstscore{k}(find(tstscore{k}~=scores(i)));              % Score of known nontargets
    N = length(tstscore{k});                                                % kn_nontgt_scr will be empty if
    pi0 = 0.5;
    pi1 = 0.5/(2*(N));
    sum1 = (pi0 + N*pi1)*exp(scores(i));
    sum2 = pi0 + pi1*sum(exp(kn_nontgt_scr));
    c_scores(i) = log(sum1/sum2);
end
assert(all(isfinite(c_scores)))
return;

function scr = normalization(GPLDAModel, gplda_scr, tst_w, testutt, normType, normp, tgt_num)
% Create a hash table containing arrays of doubles [mu sigma] that have
% been found in previous test sessions.
% Use the session name as the key. <session_name,[mu sigma]>global sessionhash;
sessionhash = java.util.Hashtable;
switch(normType)
    case {'Znorm'}
        scr = (gplda_scr - normp.znm.mu(tgt_num))./normp.znm.sigma(tgt_num);    
    case {'Tnorm'}
        if (sessionhash.containsKey(testutt) == 0),
            [mu,sigma] = comp_tnorm_para(GPLDAModel,tst_w, normp.tnm_w);
            sessionhash.put(testutt,[mu sigma]);
        else
            tnmpara = sessionhash.get(testutt); % Java methods can return array of double
            mu = tnmpara(1); 
            sigma = tnmpara(2);
        end
        scr = (gplda_scr - mu)/sigma;
    case {'ZTnorm1'}
        gplda_scr = (gplda_scr - normp.znm.mu(tgt_num))./normp.znm.sigma(tgt_num);    
        if (sessionhash.containsKey(testutt) == 0),
            [mu,sigma] = comp_tnorm_para(GPLDAModel,tst_w, normp.tnm_w);
            sessionhash.put(testutt,[mu sigma]);
        else
            tnmpara = sessionhash.get(testutt); % Java methods can return array of double
            mu = tnmpara(1); 
            sigma = tnmpara(2);
        end
        scr = (gplda_scr - mu)/sigma;
    case {'ZTnorm2'}
        gplda_scr = (gplda_scr - normp.znm.mu(tgt_num))./normp.znm.sigma(tgt_num);    
        if (sessionhash.containsKey(testutt) == 0),
            [mu,sigma] = comp_ztnorm_para(GPLDAModel,tst_w, normp.tnm_w, normp.ztnm);
            sessionhash.put(testutt,[mu sigma]);
        else
            ztnmpara = sessionhash.get(testutt); % Java methods can return array of double
            mu = ztnmpara(1); 
            sigma = ztnmpara(2);
        end
        scr = (gplda_scr - mu)/sigma;
    case {'None'}
        scr = gplda_scr;
end



function [mu,sigma] = comp_tnorm_para(GPLDAModel,tst_w, tnm_w)
N = size(tnm_w,1);
scores = zeros(N,1);
for i=1:N,
    scores(i) = PLDA_Scoring(GPLDAModel,tst_w',tnm_w(i,:)');
end
mu = mean(scores);
sigma = std(scores);


function [mu,sigma] = comp_ztnorm_para(GPLDAModel,tst_w, tnm_w, ztnm)
N = size(tnm_w,1);
scores_z = zeros(N,1);
for k=1:N,
    scores_z(k) = (PLDA_Scoring(GPLDAModel,tst_w',tnm_w(k,:)')-ztnm.mu(k))/ztnm.sigma(k);
end
mu = mean(scores_z);
sigma = std(scores_z);






