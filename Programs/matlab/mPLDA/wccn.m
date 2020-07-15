function [B, meanVec] = wccn(X, spk_logical)
% Compute WCCN projection matrix and glogal mean i-vector (col vec)
% X contains i-vectors in columns

% Get zero mean vectors
meanVec = mean(X,2);

% Estmate projection matrix B
[~, ~, spk_ids]=unique(spk_logical);    % spk_ids contains indexes to unique speakers
n_spks = max(unique(spk_ids));
dim = size(X,1);
Ws = zeros(dim,dim);
for ii = unique(spk_ids)'
    spk_sessions = find(spk_ids == ii);
    Ws = Ws + cov(X(:,spk_sessions)',1); % *length(spk_sessions); % Multiplied by length get better result
end
Winv = inv(Ws/n_spks);
B = chol(Winv,'lower');             % Compute projection matrix so that BB'=inv(W)

% Perform WCCN
% Xwccn = w*B;
return;
