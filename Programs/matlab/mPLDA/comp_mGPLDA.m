% function mGPLDAModel = comp_mGPLDA(X, L, spk_logical, GPLDA_file, n_ev, n_mix)
% Train an SNR-dependent mixture GPLDA model based on training i-vectors and speaker session info.
% This file implements the mPLDA model in my Interspeech14 paper.
% Use the GPLDA package in ~/so/Matlab/BayesPLDA
% Input:
%   X            - training ivectors in rows
%   L            - Either utterance length or SNR for each row in X
%   spk_logical  - speaker session info (BUT JFA package)
%   n_ev         - Dim of speaker space (No. of cols. in GPLDAModel.F)
%   GPLDA_file   - .mat file storing the GPLDA model structure (output)
%   n_mix        - No. of mixtures in GPLDA model
% Output:
%   mGPLDAModel  - Structure containing mGPLDA model. It has the following fields
%      pi        * Mixture weights
%      mu        * Cell array containing nMix (D x 1) mean vectors
%      W         * Cell array containing nMix (D x M) factor loading matrices
%      Sigma     * D x D diagonal covariance matrix of noise e
%      P,Q       * Cell array containing P and Q matrix (for future use)
%      const     * const for computing log-likelihood during scoring (for future use)
%      Z         * M x T common factors (one column for each z_i)
% Example:
%   Use clean_tel + clean_mic + 15dB_tel + 6dB_tel for training
%   tgtcl = load('mat/fw60/male_target-mix_mix_t500_w_1024c.mat');
%   tgt06 = load('mat/fw60/male_target-tel-06dB_mix_t500_w_1024c.mat');
%   tgt15 = load('mat/fw60/male_target-tel-15dB_mix_t500_w_1024c.mat');
%   snrcl = load('../../snr/male/male_target-mix_stats.snr');
%   snr06 = load('../../snr/male/male_target-tel-06dB_stats.snr');
%   snr15 = load('../../snr/male/male_target-tel-15dB_stats.snr');
%   X = [tgtcl.w; tgt15.w; tgt06.w]; 
%   L = [snrcl; snr15; snr06];
%   spk_logical = [tgtcl.spk_logical; tgt15.spk_logical; tgt06.spk_logical];
%   GPLDAModel = comp_mGPLDA(X, L, spk_logical, 'mat/fw60/male_mix_t500_mgplda_cln-15dB-06dB_1024c.mat', 150, 3);
% Author: M.W. Mak
% Date: Jan 2014
%   
function GPLDAModel = comp_mGPLDA(X, L, spk_logical, GPLDA_file, n_ev, n_mix, mPLDA_trainfunc)

% Set up path to use GPLDA package
if (strcmp(computer,'PCWIN')==1 || strcmp(computer,'PCWIN64')==1),    
    addpath 'D:/so/Matlab/PLDA/BayesPLDA';  
else
    addpath '~/so/Matlab/PLDA/BayesPLDA';   
end    

if ~exist('mPLDA_trainfunc','var'),
    mPLDA_trainfunc = @mPLDA_train4;              % Default training function   
end      

N_ITER = 4;
N_SPK_FAC = n_ev;

% Remove i-vecs with big norm
[X, L, spk_logical] = remove_bad_ivec(X, L, spk_logical, 40); 

% Remove speaker with less than 2 utts
[X, L, spk_logical] = remove_bad_spks(X, L, spk_logical, 2);

% Limit the number of speakers and no. of sessions per speakers 
% (for finding the relationship between the no. of speakers in PLDA and performacne)
%[X, L, spk_logical] = limit_spks(X, L, spk_logical, 200, 6);

% Compute WCCN projection matrix and global mean vector. WCNN+lennorm get the best result
X = X';     % Convert to column vectors
[projmat1, meanVec1] = wccn(X, spk_logical);

% Transform i-vector by WCCN projection matrix
X = projmat1'*(X - repmat(meanVec1,1,size(X,2)));

% Perform length normalization
Xln = (len_norm(X'))';

% LDA+WCCN on length-normalized i-vecs
[Xln, projmat2] = ldawccn(Xln', spk_logical, 200); Xln = Xln';

% Convert BUT's speaker id info (spk_logical) to PLDA equivalent (GaussPLDA)
PLDA_spkid = BUT2PLDA(spk_logical);

% Train mPLDA model using the function specified in the function handle
GPLDAModel = mPLDA_trainfunc(Xln, L, PLDA_spkid, N_ITER, N_SPK_FAC, n_mix);

GPLDAModel.projmat1 = projmat1;
GPLDAModel.projmat2 = projmat2;
GPLDAModel.meanVec1 = meanVec1;  % col vector for consistence with projmat and F

% Save mGPLDA model
fprintf('Saving mGPLDA model to %s\n',GPLDA_file);
save(GPLDA_file, 'GPLDAModel');


% comp_mGPLDA.m end

%% Private functions

% Remove i-vecs with big norm
function [X, L, spk_logical] = remove_bad_ivec(X, L, spk_logical, normlimit)
N = length(spk_logical);
normX = zeros(N,1);
for i=1:size(L,1), 
    normX(i) = norm(X(i,:)); 
end
idx = find(normX < normlimit);
X = X(idx,:);
spk_logical = spk_logical(idx);
L = L(idx);
return;

% Remove speaker with small number of utts
function [X, L, spk_logical] = remove_bad_spks(X, L, spk_logical, min_num_utts)
[~, ~, spk_ids]=unique(spk_logical);    % spk_ids contains indexes to unique speakers
numSpks = length(unique(spk_ids));
rm_idx = [];
for i=1:numSpks,
    idx = find(spk_ids == i);
    if (length(idx) < min_num_utts),
        rm_idx = [rm_idx; idx];
    end
end
spk_logical(rm_idx) = [];
X(rm_idx,:) = [];
L(rm_idx) = [];

% Limit the number of speakers
% Remove speaker with small number of utts
function [X1, L1, spk_logical1] = limit_spks(X, L, spk_logical, max_num_spks, max_num_utts)
[~, ~, spk_ids]=unique(spk_logical);    % spk_ids contains indexes to unique speakers
X1 = X;
L1 = L;
spk_logical1 = spk_logical;
numSpks = length(unique(spk_ids));
if (numSpks > max_num_spks)
    rm_idx = [];
    for i=max_num_spks+1:numSpks,
        idx = find(spk_ids == i);
        rm_idx = [rm_idx; idx];
    end
    spk_logical1(rm_idx) = [];
    X1(rm_idx,:) = [];
    L1(rm_idx) = [];
end;
[~, ~, spk_ids1]=unique(spk_logical1);
numSpks1 = length(unique(spk_ids1));
rm_idx = [];
for i=1:numSpks1,
    idx = find(spk_ids1 == i);
    if (length(idx) > max_num_utts)
        rm_idx = [rm_idx; idx(max_num_utts+1:end)];
    end
end
spk_logical1(rm_idx) = [];
X1(rm_idx,:) = [];
L1(rm_idx) = [];

