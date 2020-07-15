% Compute the GPLDA model based on training i-vectors and speaker session info.
% Use the GPLDA package in ~/so/Matlab/BayesPLDA
% Input:
%   w            - training ivectors in rows
%   spk_logical  - speaker session info (BUT JFA package)
%   n_ev         - Dim of speaker space (No. of cols. in GPLDAModel.F)
%   GPLDA_file   - .mat file storing the GPLDA model structure
% Output:
%   GPLDAModel   - Structure containing GPLDA model. It has the following fields
%     projmat1   * Whitening or WCCN projection matrix (must be done before length norm)
%     meanVec1   * Global mean of training ivectors before lennorm
%     projmat2   * LDA + WCCN projection matrix (must be done after length norm)
%     meanVec2   * Global mean of training ivectors after Whitening+lennorm+LDA+WCCN
%     V          * Speaker factor loading matrix
%     Sigma      * Full covariance of residual
%     P,Q        * Pre-computed matrices for GPLDA scoring
% Example:
%     mix = load('mat/fw60/male_target-mix_mix_t500_w_1024c.mat');
%     GPLDAModel=comp_GPLDA1(mix.w, mix.spk_logical, 'mat/fw60/male_mix_t500_gplda_clean_1024c.mat', 150);
%     GPLDAModel=comp_GPLDA1(mix.w, mix.spk_logical, 'mat/fw60/temp.mat', 150);
% Author: M.W. Mak
% Date: July 2012
%   
function GPLDAModel = comp_GPLDA1(w, spk_logical, GPLDA_file, n_ev)

% Set up path to use GPLDA package. 
% Note: Your GPLDA package may be in a different folder
if (strcmp(computer,'PCWIN')==1 || strcmp(computer,'PCWIN64')==1),    
    addpath 'D:/so/Matlab/PLDA/BayesPLDA';  
else
    addpath '~/so/Matlab/PLDA/BayesPLDA';
end    

N_ITER = 10;
N_SPK_FAC = n_ev;

% Remove i-vecs with big norm
[w, spk_logical] = remove_bad_ivec(w, spk_logical, 40); 

% Remove speaker with less than 2 utts
[w, spk_logical] = remove_bad_spks(w, spk_logical, 2);

% Limit the number of speakers and no. of sessions per speakers 
% for finding the relationship between the no. of speakers in PLDA and performacne
%[w, spk_logical] = limit_spks(w, spk_logical, 400, 5);

% Estimate WCCN matrix for whitening. WCNN+lennorm get the best result
X = w';     % Convert to column vectors
[projmat1, meanVec1] = wccn(X, spk_logical);

% Transform i-vector by WCCN projection matrix (whitening)
X = projmat1'*(X - repmat(meanVec1,1,size(X,2)));

% Perform length normalization
X = (len_norm(X'))';

% Perform LDA+WCCN on length-normalized i-vecs
[X, GPLDAModel.projmat2] = ldawccn(X', spk_logical, 200); 
X = X';

% Subtract the mean so that the scoring function becomes independent of global mean
meanVec2 = mean(X,2);
X = X - repmat(meanVec2, 1, size(X,2));

% Convert BUT's speaker id format to PLDA format
PLDA_spkid = BUT2PLDA(spk_logical);

% Train GPLDA model without eigenchannel
[V,Sigma,P,Q,const,Z] = PLDA_train1(X, PLDA_spkid, N_ITER, N_SPK_FAC);

% Construct GPLDA model structure
GPLDAModel.projmat1 = projmat1;
GPLDAModel.meanVec1 = meanVec1;  % col vector for consistence with projmat and F
GPLDAModel.V = V;
GPLDAModel.Sigma = Sigma;
GPLDAModel.P = P;
GPLDAModel.Q = Q;
GPLDAModel.const = const;
GPLDAModel.Z = Z;
GPLDAModel.meanVec2 = meanVec2;

% Save PLDA model
fprintf('Saving GPLDA model to %s\n',GPLDA_file);
save(GPLDA_file, 'GPLDAModel');

% comp_GPLDA.m end

%% Private functions

% Remove i-vecs with big norm
function [w, spk_logical] = remove_bad_ivec(w, spk_logical, normlimit)
N = length(spk_logical);
normw = zeros(N,1);
for i=1:size(w,1), 
    normw(i) = norm(w(i,:)); 
end
idx = find(normw < normlimit);
w = w(idx,:);
spk_logical = spk_logical(idx);
return;

% Remove speaker with small number of utts
function [w, spk_logical] = remove_bad_spks(w, spk_logical, min_num_utts)
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
w(rm_idx,:) = [];

% Limit the number of speakers
% Remove speaker with small number of utts
function [w1, spk_logical1] = limit_spks(w, spk_logical, max_num_spks, max_num_utts)
[~, ~, spk_ids]=unique(spk_logical);    % spk_ids contains indexes to unique speakers
w1 = w;
spk_logical1 = spk_logical;
numSpks = length(unique(spk_ids));
if (numSpks > max_num_spks)
    rm_idx = [];
    for i=max_num_spks+1:numSpks,
        idx = find(spk_ids == i);
        rm_idx = [rm_idx; idx];
    end
    spk_logical1(rm_idx) = [];
    w1(rm_idx,:) = [];
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
w1(rm_idx,:) = [];

    


