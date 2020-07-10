function [N, F, S] = collect_suf_stats(data, m, v, w)
% Collect sufficient stats in baum-welch fashion
%  [N, F, S] = collect_suf_stats(data, M, V, W) returns the vectors N and
%  F of zero- and first- order statistics, respectively, where data is a 
%  dim x nFrames matrix of features, M is dim x gaussians matrix of GMM means
%  V is a dim x gaussians matrix of GMM variances, W is a vector of GMM weights.

n_mixtures  = size(w, 1);
dim         = size(m, 1);

% compute the GMM posteriors for the given data
gammas = gaussian_posteriors(data, m, v, w);

% zero order stats for each Gaussian are just sum of the posteriors (soft
% counts)
N = sum(gammas,2);      % N_i = sum_t gamma_i(t)

% first order stats is just a (posterior) weighted sum
F = data * gammas';     % F_ij = sum_t gamma_i(t)x_tj   i=1:nmix; j=1:dim 
F = reshape(F, n_mixtures*dim, 1);

% Compute 2nd order stats. Comment out to save computation
S = [];
%num_frms = size(data,2);
%S = zeros(dim,n_mixtures);
% for c = 1:n_mixtures,
%     mTmp = repmat(m(:,c),1,num_frms);
%     S(:,c) = ((data-mTmp).^2) * gammas(c,:)';     % Consider the diag elements of (x-mu)(x-mu)'
% end
% S = reshape(S, n_mixtures*dim, 1);
return;

%%
% A clearer version of computing G but slower
% for c = 1:n_mixtures,
%     Sxxc = zeros(dim,1);
%     for t = 1:num_frms,
%         tmp = data(:,t)-m(:,c);
%         Sxxc = Sxxc + gammas(c,t)*tmp.^2;
%     end
%     G = G + N(c)*log(1/const*prod(v(:,c))^0.5) - 0.5*sum(Sxxc./v(:,c)); 
% end

    