% Compute the scores of PLDA model, given a set of target speaker i-vec and test i-vec
%
% Input:
%   target_w_file        - Cell array containing n i-vectors files 
%   tnorm_w_file         - Remain here for backward compatibility. Should be empty
%   test_w_file          - File containing test i-vectors
%   ndx_lstfile          - List file specifying the evaluation trials
%   znorm_para_file      - Remain here for backward compatibility. Should be empty
%   ztnorm_para_file     - Remain here for backward compatibility. Should be empty
%   normType             - Remain here for backward compatibility. Should be 'None'
%   evlfile              - Output file in NIST SRE format
%   GPLDA_file           - Cell array containing the Gaussian PLDA model files corresponding to the target files in target_w_file{}
%   opt                  - Optional parameters controlling the behaviour of the scoring process
% Example:
%   score_gplda_w({'mat/fw60/male_target-tel_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-15dB_mix_t500_w_1024c.mat','mat/fw60/male_target-tel-06dB_mix_t500_w_1024c.mat'},'','mat/fw60/male_test-tel-phn_mix_t500_w_1024c.mat','lst/fw60/male_ndx_cc4_stats.lst','','','None','evl/fw60/gplda60_male_cc4_1024c.evl','mat/fw60/male_mix_t500_gplda_noisy_1024c.mat')
% Author: M.W. Mak
% Date: 24 Oct. 2012
%
function scores = score_gplda_w(target_w_file,tnorm_w_file,test_w_file,ndx_lstfile,...
                                znorm_para_file, ztnorm_para_file, normType,...
                                evlfile, GPLDA_file, opt)
                            
% opt.mode = 'scravg': Scoring test ivec against target ivecs individually (default)
% opt.mode = 'ivcavg': Scoring test ivec against the averaged i-vecs within each SNR group without average the SNR
% opt.mode = 'bybook': By-the-book scoring, i.e., computing the truth LLR using all target ivecs
if ~exist('opt','var'),
    opt.mode = 'scravg';          
end 
                            
if nargin ~= 9 && nargin ~=10,
    disp('Need 9 or 10 input arguments');
    return;
end

% Set up path to use GPLDA package
if (strcmp(computer,'PCWIN')==1 || strcmp(computer,'PCWIN64')==1),
    addpath 'D:/so/Matlab/PLDA/BayesPLDA';  
else
    addpath '~/so/Matlab/PLDA/BayesPLDA';
end    

% Load GPLDA model structure to obtain the structure GPLDAModel
disp(['Loading ' GPLDA_file]);
gplda = load(GPLDA_file);
GPLDAModel = gplda.GPLDAModel;

% Load i-vectors of target speakers.
% Concatenate the i-vectors, spk_logical, spk_physical, num_frames in the input target_w_files
trn = cell(length(target_w_file)); 
tgt = struct('w',[],'spk_logical',[],'spk_physical',[],'num_frames',[]);
for i=1:length(target_w_file),
    disp(['Loading ' target_w_file{i}]);
    trn{i} = load(target_w_file{i});       % Load w and spk_logical of target speakers
    tgt.w = [tgt.w; trn{i}.w];
    tgt.spk_logical = [tgt.spk_logical; trn{i}.spk_logical];
    tgt.spk_physical = [tgt.spk_physical; trn{i}.spk_physical];
    tgt.num_frames = [tgt.num_frames; trn{i}.num_frames];
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
mode = opt.mode;
fprintf('Scoring mode: %s\n', mode);
num_tests = numlines(ndx_lstfile);
scores = zeros(num_tests,1);
ndx.spk_logical = parse_list(ndx_lstfile);
n_tstutt = length(tst.spk_logical);
C_target = cell(n_tstutt,1);
C_testutt = cell(n_tstutt,1);
C_channel = cell(n_tstutt,1);

% To speed up scoring, we transform the i-vecs of all target and test ivecs first
% For ivec averaging mode, it is better to average the 500-dim i-vectors before performing i-vec preprocessing.
switch(mode)
    case {'scravg','bybook'}
        tgt.w = (preprocess_ivecs(GPLDAModel, tgt.w'))';
        tst.w = (preprocess_ivecs(GPLDAModel, tst.w'))';
    case {'ivcavg'}
        tst.w = (preprocess_ivecs(GPLDAModel, tst.w'))';
end

% To speed up, avoid using indexed array
ndx_spk_logical = ndx.spk_logical;
tgt_spk_logical = tgt.spk_logical;

tic;
for i=1:num_tests,                           % For each trial
    session_name = ndx_spk_logical{i};       % Test session name
    field = rsplit(':',session_name);        % Split spk_logical into target and test utt (e.g. 100396:tabfsa_sre12_B)
    target = field{1};
    testutt = field{2};
    field = rsplit('_',testutt);
    channel = lower(field{end});
        
    % Find the index of the test utt
    k = find(strncmp(testutt, tst.spk_logical,length(testutt))==1); 
    tst_w = tst.w(k,:);                              % k should be a scalar 
   
    % Find target session of the current target speaker
    tgt_sessions = find(strncmp(target, tgt_spk_logical,length(target))==1);  % Find out against which target the utt is tested

    % Exclude short target utterances
    n_tgt_frms = tgt.num_frames(tgt_sessions);
    tgt_ss = tgt_sessions(n_tgt_frms>=1000);
    if (isempty(tgt_ss)~=1)
        tgt_sessions = tgt_ss;
    end                

    % Make sure that target sessions exist
    assert(~isempty(tgt_sessions),sprintf('Missing sessions of %s in score_gplda_w.m',target));    
    
    % Get i-vectors of target's training sessions. 
    tgt_w = tgt.w(tgt_sessions,:);
        
    % Compute the scores of current tst utt against all training utts of the selected target speaker
    % Reject this speaker if the speaker does not have enrollment utterances
    if (isempty(tgt_sessions)~=1),
        switch(mode)
            case {'scravg'}
                gplda_scr = PLDA_GroupScoring(GPLDAModel, tgt_w', tst_w');
            case {'ivcavg'}
                tgt_w = mean(tgt_w,1);
                tgt_w = (preprocess_ivecs(GPLDAModel, tgt_w'))';
                gplda_scr = PLDA_GroupScoring(GPLDAModel, tgt_w', tst_w');
            case {'bybook'}
                gplda_scr = PLDA_BybookScoring(GPLDAModel, tgt_w', tst_w');
            otherwise
                disp('Invalid opt.mode parameter');
        end
    else
        gplda_scr = -174; % Should not reach here if the line assert() above is in effect.
    end
    
    % Perform score normalization (if necessary) and compute the mean PLDA score
    if (strcmp(normType,'None')==1),
        scores(i) = mean(gplda_scr);
    else
        tgt_num = find(strcmp(target, normp.znm.spk_id)==1);            % Find the target number (index in normp)
        scores(i) = mean(normalization(GPLDAModel, gplda_scr, tst_w, testutt, normType, normp, tgt_num));
    end
    
    % Show scoring progress
    if mod(i-1,10000)==0,
        fprintf('(%d/%d) %s,%s: %f\n',i,num_tests,target,testutt,scores(i));
    end
    
    % Copy target, testutt and channel to cell array for saving to file later
    C_target{i} = target;
    C_testutt{i} = testutt;
    C_channel{i} = channel;     
end
toc;

% Save the score to .evl file
fp = fopen(evlfile,'w');
for i=1:num_tests,
    testsegfile = rsplit('_A|_B',C_testutt{i});
    testsegfile{1} = strcat(testsegfile{1},'.sph');
    fprintf(fp,'%s,%s,%c,%.7f\n', C_target{i}, testsegfile{1}, C_channel{i}, scores(i));    
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




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     lenRatio = trn{idx}.num_frames(tgt_sessions)/tst.num_frames(k);
%     maxRatio = 15; minRatio = 1/maxRatio;
%     idx = find(lenRatio>=minRatio & lenRatio<=maxRatio);
%     if isempty(idx) == 1,
%         idx = 1:length(lenRatio);
%     end
%     gplda_scr = zeros(length(idx),1);
%     for j=1:length(idx),               
%        gplda_scr(j) = PLDA_Scoring(GPLDAModel,tst_w',trn_w(idx(j),:)'); % Score test i-vec against each of target's i-vecs
%     end

