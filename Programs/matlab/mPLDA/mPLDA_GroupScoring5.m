%=========================================================
% function [scores, clusterID] = mPLDA_GroupScoring1(mPLDAModel, Xs, xt)
% Implement the SNR-independent mixture of PLDA scoring. This function should
% work with mPLDA_train5.m
%
% This function is for opt.mode = 'scravg' and 'ivcsnravg' in snr3_score_gplda_w.m.
% When mode = 'icvsnravg', the SNR within the same group must be averaged and the
% length of Ls must be equal to the number of SNR groups.
% 
%   Input:
%       mPLDAModel     - mPLDA model structure
%       Xs             - Matrix containing a set of column i-vectors of speaker s
%       xt             - Second un-normalized i-vector (column vec)
%   Output:
%       scores         - PLDA scores (unnormalized) of Xs and xt
% Author: M.W. Mak
% Date: Aug 2015
%=========================================================
function scores = mPLDA_GroupScoring5(mPLDAModel, Xs, xt)

% Whitening/WCCN then length normalization
n_vecs = size(Xs,2);
Xs = mPLDAModel.projmat1' * (Xs-repmat(mPLDAModel.meanVec1,1,n_vecs));
xt = mPLDAModel.projmat1' * (xt-mPLDAModel.meanVec1);
Xs = (len_norm(Xs'))';
xt = (len_norm(xt'))';
Xs = mPLDAModel.projmat2' * Xs;
xt = mPLDAModel.projmat2' * xt;

% Extract paras from model structure
varphi = mPLDAModel.varphi;
m = mPLDAModel.m;
Icov = mPLDAModel.Icov;
Icov2 = mPLDAModel.Icov2;
logDetCov = mPLDAModel.logDetCov;
logDetCov2 = mPLDAModel.logDetCov2;
V = mPLDAModel.V;
K = length(varphi);

% Precompute likelihood of test i-vector for speed
sum3 = 0;
expterm = zeros(K,1);
for k = 1:K,
    detterm = 0.5*logDetCov{k};
    expterm(k) = exp(-0.5*Mahaldist(xt, m{k}, Icov{k})-detterm); 
    sum3 = sum3 + varphi(k)*expterm(k);
end
assert(sum3>0,'Divided by zero (sum3) in mPLDA_GroupScoring.m');

% Compute log-likelihood score
scores = zeros(size(Xs,2),1);
for s = 1:length(scores),
    sum1 = 0; sum2 = 0; 
    xs = Xs(:,s);
    for ks = 1:K,
        for kt = 1:K,
            sum1 = sum1 + varphi(ks)*varphi(kt) * ...
                   exp(-0.5*Mahaldist([xs; xt],[m{ks}; m{kt}],Icov2{ks,kt})-0.5*logDetCov2{ks,kt});
        end
    end
    for k = 1:K,
        detterm = 0.5*logDetCov{k};
        sum2 = sum2 + varphi(k)*exp(-0.5*Mahaldist(xs, m{k}, Icov{k})-detterm); 
    end
    assert(sum1>0,'Sum1 is 0 in mPLDA_GroupScoring.m');
    assert(sum2>0,'Divided by zero (sum2) in mPLDA_GroupScoring.m');
    scores(s) = log(sum1) - (log(sum2)+log(sum3));
end
    
    

%% Private functions

% Return the Mahalanobis distance between x and mu with covariance Sigma
% Both x and mu are col vectors
function md = Mahaldist(x, mu, Icov)
temp = x - mu;
md = (temp'*Icov)*temp;




        