%=========================================================
% function [scores, clusterID] = mPLDA_GroupScoring1(mPLDAModel, Xs, Ls, xt, lt)
% Implement the mixture of PLDA scoring in 
% M.W. Mak, SNR-Dependent Mixture of PLDA for Noise Robust Speaker Veriﬁcation, Intespeech2014.
% The test i-vector is scored against each of the target-speaker's i-vectors. 
% No averaging is applied to the target-speaker's i-vectors.
% This function is for opt.mode = 'scravg' and 'ivcsnravg' in snr3_score_gplda_w.m.
% When mode = 'icvsnravg', the SNR within the same group must be averaged and the
% length of Ls must be equal to the number of SNR groups.
% 
%   Input:
%       mPLDAModel     - mPLDA model structure
%       Xs             - Matrix containing a set of column i-vectors of speaker s
%       xt             - Second un-normalized i-vector (column vec)
%       Ls             - Length or SNR of utterances in Xs
%       lt             - Length or SNR of utterance in xt
%   Output:
%       scores         - PLDA scores (unnormalized) of Xs and xt
% Author: M.W. Mak
% Date: June 2014
% Update May 2015 Mak: The likelihood of test i-vector is computed outside onces only,
%                      which speed up the computation of LLR when the number of target
%                      speaker i-vecs is large.
%=========================================================
function scores = mPLDA_GroupScoring1(mPLDAModel, Xs, xt, Ls, lt)

% Whitening/WCCN then length normalization
n_vecs = size(Xs,2);
Xs = mPLDAModel.projmat1' * (Xs-repmat(mPLDAModel.meanVec1,1,n_vecs));
xt = mPLDAModel.projmat1' * (xt-mPLDAModel.meanVec1);
Xs = (len_norm(Xs'))';
xt = (len_norm(xt'))';
Xs = mPLDAModel.projmat2' * Xs;
xt = mPLDAModel.projmat2' * xt;

% Extract paras from model structure
pi = mPLDAModel.pi;
mu = mPLDAModel.mu;
sigma = mPLDAModel.sigma;
m = mPLDAModel.m;
Icov = mPLDAModel.Icov;
Icov2 = mPLDAModel.Icov2;
logDetCov = mPLDAModel.logDetCov;
logDetCov2 = mPLDAModel.logDetCov2;
K = length(pi);

% Precompute likelihood of test i-vector for speed
posty_lt = posterior1G(lt, pi, mu, sigma);
sum3 = 0;
expterm = zeros(K,1);
for k = 1:K,
    detterm = 0.5*logDetCov{k};
    expterm(k) = exp(-0.5*Mahaldist(xt, m{k}, Icov{k})-detterm); 
    sum3 = sum3 + posty_lt(k)*expterm(k);
end
assert(sum3>0,'Divided by zero (sum3) in mPLDA_GroupScoring.m');

% Compute log-likelihood score
scores = zeros(length(Ls),1);
for s = 1:length(Ls),
    sum1 = 0; sum2 = 0; 
    xs = Xs(:,s);
    ls = Ls(s);
    posty = posterior2G(ls, lt, pi, mu, sigma);
    for ks = 1:K,
        for kt = 1:K,
            sum1 = sum1 + posty(ks, kt) * exp(-0.5*Mahaldist([xs; xt],[m{ks}; m{kt}],Icov2{ks,kt})-0.5*logDetCov2{ks,kt});
        end
    end
    posty_ls = posterior1G(ls, pi, mu, sigma);
    for k = 1:K,
        detterm = 0.5*logDetCov{k};
        sum2 = sum2 + posty_ls(k)*exp(-0.5*Mahaldist(xs, m{k}, Icov{k})-detterm); 
    end
    assert(sum1>0,'Sum1 is 0 in mPLDA_GroupScoring1.m');
    assert(sum2>0,'Divided by zero (sum2) in mPLDA_GroupScoring1.m');
    scores(s) = log(sum1) - (log(sum2)+log(sum3));
end
    
    
%% Private functions

% Posterior of single SNR
function posty = posterior_y1(l, pi, mu, sigma)
K = length(pi);
posty = ones(K,1);
wlh = zeros(1,K);
for r = 1:K,
    wlh(r) = pi(r)*mvnpdf(l,mu(r),sigma(r)^2);
end
temp = sum(wlh);
assert(~isnan(temp),'Assertion in posteror_y1: sum(wlh) is NaN');
for k = 1:K,
    posty(k) = wlh(k)/temp;
end

% Posterior of 2-SNR
% Note: mvnpdf is very slow. We can speedup computation by writing our own function
% where the det(sigma2) is pre-computed.
function posty = posterior_y2(ls, lt, pi, mu, sigma)
K = length(pi);
wlh = zeros(K,K);
for p = 1:K,
    for q = 1:K,
        mu2 = [mu(p) mu(q)];
        sigma2 = diag([sigma(p)^2 sigma(q)^2]);
        wlh(p,q) = pi(p)*pi(q)*mvnpdf([ls lt], mu2, sigma2);
    end
end
temp = sum(sum(wlh));
assert(temp>0,'Divided by zeros in posterior_y2 of mPLDA_GroupScoring1.m');
posty = wlh/temp; 

% Return the Mahalanobis distance between x and mu with covariance Sigma
% Both x and mu are col vectors
function md = Mahaldist(x, mu, Icov)
temp = x - mu;
md = (temp'*Icov)*temp;




        