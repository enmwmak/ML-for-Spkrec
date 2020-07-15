%=========================================================
% function scores = mPLDA_fastGroupScoring1(mPLDAModel, Xs, Ls, xt, lt)
% Implement fast scoring of the mixture of PLDA scoring in 
% M.W. Mak, SNR-Dependent Mixture of PLDA for Noise Robust Speaker Veriﬁcation, Intespeech2014.
% The test i-vector is scored against each of the target-speaker's i-vectors. 
% No averaging is applied to the target-speaker's i-vectors.
% This function is for opt.mode = 'scravg' and 'ivcsnravg' in snr3_score_gplda_w.m.
% When mode = 'icvsnravg', the SNR within the same group must be averaged and the
% length of Ls must be equal to the number of SNR groups.
% 
% Note: To achieve fast scoring, only the Gaussian with largest posterior will be considered.
%   Input:
%       mPLDAModel     - mPLDA model structure
%       Xs             - Matrix containing a set of column i-vectors of speaker s
%       xt             - Second un-normalized i-vector (column vec)
%       Ls             - Length of utterances in Xs
%       lt             - Length of utterances in xt
%   Output:
%       scores         - PLDA scores (unnormalized) of Xs and xt
% Author: M.W. Mak
% Date: June 2014
% Update May 2015 Mak: The likelihood of test i-vector is computed outside onces only,
%                      which speed up the computation of LLR when the number of target
%                      speaker i-vecs is large.
%=========================================================
function scores = mPLDA_fastGroupScoring1(mPLDAModel, Xs, xt, Ls, lt)

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

% Find the index to the maximum posterior test SNR
posty_lt = posterior1G(lt, pi, mu, sigma);
[~,kt1] = max(posty_lt);

% Compute log-likelihood score
scores = zeros(length(Ls),1);
for s = 1:length(Ls),
    xs = Xs(:,s);
    ls = Ls(s);
    posty = posterior2G(ls, lt, pi, mu, sigma);
    [ks2, kt2] = find(posty==max(posty(:)));
    posty_ls = posterior1G(ls, pi, mu, sigma);
    [~,ks1] = max(posty_ls);
    Pst = mPLDAModel.P{ks2,kt2};
    Qst = mPLDAModel.Q{ks2,kt2};
    Qts = mPLDAModel.Q{kt2,ks2};
    ms = mPLDAModel.m{ks2};
    mt = mPLDAModel.m{kt2};
    const = mPLDAModel.const(ks2,kt2);
    scores(s) = log(posty(ks2,kt2)) - log(posty_ls(ks1)) - log(posty_lt(kt1)) ...
                + 0.5*xs'*Qst*(xs + 2*ms) + 0.5*xt'*Qts*(xt + 2*mt) ...
                + xs'*Pst*(xt + mt) + xt'*Pst'*ms ...
                + const;
end

% Fast scoring without using P and Q
% pi = mPLDAModel.pi;
% mu = mPLDAModel.mu;
% sigma = mPLDAModel.sigma;
% m = mPLDAModel.m;
% Icov = mPLDAModel.Icov;
% Icov2 = mPLDAModel.Icov2;
% logDetCov = mPLDAModel.logDetCov;
% logDetCov2 = mPLDAModel.logDetCov2;
% scores = zeros(length(Ls),1);
% for s = 1:length(Ls),
%     xs = Xs(:,s);
%     ls = Ls(s);
%     posty = posterior2G(ls, lt, pi, mu, sigma);
%     [ks2, kt2] = find(posty==max(posty(:)));
%     posty_ls = posterior1G(ls, pi, mu, sigma);
%     [~,ks1] = max(posty_ls);
%     sum3 = posty_lt(kt1)*exp(-0.5*Mahaldist(xt, m{kt1}, Icov{kt1})-0.5*logDetCov{kt1});
%     sum2 = posty_ls(ks1)*exp(-0.5*Mahaldist(xs, m{ks1}, Icov{ks1})-0.5*logDetCov{ks1});
%     sum1 = posty(ks2, kt2) * exp(-0.5*Mahaldist([xs; xt],[m{ks2}; m{kt2}],Icov2{ks2,kt2})-0.5*logDetCov2{ks2,kt2});
%     scores(s) = log(sum1) - (log(sum2)+log(sum3));
% end
% 
    
%% Private functions

function md = Mahaldist(x, mu, Icov)
temp = x - mu;
md = (temp'*Icov)*temp;

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
%       where the det(sigma2) is pre-computed.
%       Another speed up is to input the 2x2 precision matrix
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
assert(temp>0,'Divided by zeros in posterior_y2 of mPLDA_GroupScoring.m');
posty = wlh/temp; 






        