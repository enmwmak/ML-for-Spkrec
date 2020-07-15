function [X,projmat] = ldawccn(w, spk_logical, n_ev)
% Estimate the LDA+WCCN matrix  from w and spk_logical. 

% Rank of WCCN and LDA
% S_w = sum_s sum_i (x_si – m_s)(x_si – m_s)’
% Rank(S_w) = S * rank(cov(x_si))     if rank(cov(x_si) > 1
% Rank(S_w) = S                       if rank(cov(x_si) = 1


[~, ~, spk_ids]=unique(spk_logical);        % spk_ids contains indexes to unique speakers
nf = size(w,2);                             % Dim of i-vec
Sw = zeros(nf,nf);                          % Within-spk cov
Sb = zeros(nf,nf);                          % Between-spk cov
mu = mean(w,1);                             % Glogal mean of total factors
for ii = unique(spk_ids)'
    spk_sessions = spk_ids == ii;           % Sessions indexes of speaker ii
    ws = w(spk_sessions,:);
    Sw = Sw + cov(ws,1); 
    mu_s = mean(ws,1);                      % Mean of speaker-dependent total factors
    Sb = Sb + (mu_s-mu)'*(mu_s-mu);               
end

% Find the n_ev largest eigenvectors and eigenvalues of AV=Lambda BV, i.e., 
% find the eigenvectors of inv(Sw)*Sb
[V,~] = eigs(Sb,Sw,n_ev);

% Project the total factors to a reduced space using the LDA projection matrix
lda_y = w*V;

% Find WCCN
n_spks = max(unique(spk_ids));
Ws = zeros(n_ev,n_ev);
for ii = unique(spk_ids)'
    spk_sessions = find(spk_ids == ii);
    Ws = Ws + cov(lda_y(spk_sessions,:),1)*length(spk_sessions);
end
Winv = inv(Ws/n_spks);
B = chol(Winv,'lower');     % Compute projection matrix so that BB'=inv(W)


% Return projected vectors (row vecs) in X and LDA+WCCN projection matrix
projmat = V*B;
X = w*projmat;
return;
