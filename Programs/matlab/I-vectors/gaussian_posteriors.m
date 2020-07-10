function gammas = gaussian_posteriors(data, m, v, w)
% Computes Gaussian posterior probs for the given data
%   For each frame (column) of data, compute the vector of posterior probs
%   of the given GMM given by means, vars, and weights

n_mixtures  = size(w, 1);
n_frames    = size(data, 2);
dim         = size(data, 1);

% precompute the model g-consts and normalize the weights
a = w ./ (((2*pi)^(dim/2)) * sqrt(prod(v)'));

gammas = zeros(n_mixtures, n_frames); 

for ii =1:n_mixtures 
  gammas(ii,:) = gaussian_function(data, a(ii), m(:,ii), v(:,ii));
end
gamasum = sum(gammas,1);       % sum_k w_k p(x|mu_k,Sigma_k)

% normalize
gammas=bsxfun(@rdivide, gammas, gamasum);
