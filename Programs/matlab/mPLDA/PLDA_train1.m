%==============================================================================
% function [V,Sigma,P,Q,const,Z] = PLDA_train1(X, IDmat, nIters, M)
% Implement Gaussian PLDA (training) without eigenchannel
%
% FA model: x = mu + V*z + e   
% where mu is the mean of X, z is the speaker factor, and e is the noise (residue)
% 
% Note: While the program still computes the global mean, the scoring function that
%       use P and Q requires that the global mean of X is zero.
% Input:
%   X     - D x T data matrix with D-dim column vectors (zero mean)
%   M     - No. of common factors (latent variables)
%   IDmat - ID matrix (numVecs x numSpks). For each row, the column with a 1
%           means that the i-vector is produced by the corresponding speaker.
%           The rest should be zero.
% Output:
%   V     - D x M factor loading matrice
%   Sigma - D x D covariance matrix of noise e
%   P,Q   - P and Q matrix in Garcia-Romero's paper
%   const - Constant independent of i-vecs in the scoring function.
%   Z     - M x T common factors (one column for each z_i)
%
% By M.W. Mak on 7 Sept. 2012
%========================================================================
function [V,Sigma,P,Q,const,Z] = PLDA_train1(X, IDmat, nIters, M)
[D,T] = size(X);
nSpks = size(IDmat,2);

% Initialize factor loading matrix V and noise cov Sigma
% Note (Mak, 18/9/15): Apply PCA to cluster means will limit the no. of speaker factors to
% the number of speakers in IDmat. If you want to have larger number of speaker factors,
% Uncomment the line "V = randn(D,M);" so that V start from random. 
Data = double(X) - repmat(mean(X,2), 1, T);
clusterMeans = (Data * IDmat) * (diag(1 ./ (sum(IDmat)))); V =  trainPCA(clusterMeans);
%V = randn(D,M);
V = V(:,1:M);
Sigma = 0.01*eye(D);

% Prepare EM
mu = mean(X,2);
IL = cell(1,nSpks);             % Inverse of L_i
E_z = cell(1,nSpks);            % <z_ij|X>
E_zz = cell(1,nSpks);           % <z_ij z_ij'|X>

%llh_old = getLikelihood(Data, V, Sigma, zeros(M,nSpks), IDmat);
llh_old = getEvidence(Data, V, Sigma);
fprintf('llh before EM = %f\n', llh_old);

% Start PLDA EM
for t = 1:nIters,
    % E-step
    VtISig = V'/Sigma;                      % V'*inv(Sigma)
    for i=1:nSpks,
        idx = find(IDmat(:,i)==1);
        Hi = length(idx);
        IL{i} = inv(eye(M)+Hi*(VtISig*V));  % Note: Hi should * a square matrix
        E_z{i} = IL{i}*VtISig*sum(X(:,idx)-repmat(mu,1,Hi),2);
        E_zz{i} = IL{i} + E_z{i}*E_z{i}';
    end
    Z = cell2mat(E_z);
    
    % M-step
    
    % Update V
    sum1 = zeros(D,M);
    sum2 = zeros(M,M);
    for i=1:nSpks,
        idx = find(IDmat(:,i)==1);
        Hi = length(idx);
        Xs = X(:,idx)-repmat(mu,1,Hi);
        sum1 = sum1 + sum(Xs,2)*E_z{i}';
        sum2 = sum2 + Hi*E_zz{i};
    end
    V = sum1/sum2;                          % V = sum1 * inv(sum2);    

    % Update Sigma
    sum3 = zeros(D,D);
    for i=1:nSpks,
        idx = find(IDmat(:,i)==1);
        Hi = length(idx);
        Xs = X(:,idx)-repmat(mu,1,Hi);
        for j=1:Hi,
            sum3 = sum3 + Xs(:,j)*Xs(:,j)'-V*E_z{i}*Xs(:,j)';
        end
    end
    Sigma = sum3/T;
    %figure; plot(diag(Sigma));
    %Sigma may not be exactly symmetric, but it can pass the function chol.m
    %issym = @(x) isequal(tril(x), triu(x)');
    %assert(issym(Sigma)==1,'PLDA_train.m: Cov matrix not symmetric');    
    %Sigma = diag(diag(Sigma));
    
    %sum1 << sum2, so sum2 can be ignored.
    %sum1 = 0; sum2 = 0;
    %for i=1:nSpks,
    %    sum1 = sum1 - trace((VtISig*V+eye(M))*E_z{i}*E_z{i}') + E_z{i}'*(VtISig*V+eye(M))*E_z{i};
    %    sum2 = sum2 + trace((VtISig*V+eye(M))*E_zz{i});
    %end
    
    % Log-likelihood
    %llh = getLikelihood(Data, V, Sigma, Z, IDmat);
    llh = getEvidence(Data, V, Sigma);
    perinc = abs(100*(llh - llh_old)/llh_old);
    if llh > llh_old,
        fprintf('Ater iteration %d: llh = %.2f; %%increase = %.2f%%\n',t,llh,perinc);
    else
        fprintf('Ater iteration %d: llh = %.2f; %%decrease = %.2f%%\n',t,llh,perinc);
    end
    llh_old = llh;
end

% Compute P and Q matrix in Garcia-Romero's paper
[P, Q, const] = prepare_scoring(V,Sigma);
return;

%=======================================================================
% function  [princComp meanVec] = trainPCA(data)
%
% Principal Component Analysis (PCA)
%
%   Input:
%       Data    - NFeature  x NSample   Training data
%
%   Output:
%       princComp   - Principal Expectation of latent varialbe h and latent noise 
%       meanVec     - Mean vector of the data
%=======================================================================
function [princComp meanVec] = trainPCA(data)
[~, nData] = size(data);
meanVec = mean(data,2);
data = data-repmat(meanVec,1,nData);

XXT = data'*data;
[~, LSq V] = svd(XXT);
LInv = 1./sqrt(diag(LSq));
princComp  = data * V * diag(LInv);
% End of function trainPCA

%=======================================================================
% function  llh = getEvidence(Data, V, Sigma)
%
% Compute log marginal likelihood of Data: sum_i log N(x_i|0,VV'+Sigma)
%
%   Input:
%       Data    - Mean-removed training data (NFeature  x NSample)
%       V       - Factor loading matrix of latent identity variable
%       Sigma   - Covariance matrix of noise (diagonal)
%
%   Output:
%       llh     - log-likelihood
%=======================================================================
function llh = getEvidence(Data, V, Sigma)
Lambda = V*V'+Sigma;
logDetLambda = logDet(Lambda);
ILambda = inv(Lambda);
s = 0;
[dim,nVectors] = size(Data);
for i = 1:nVectors,
    x = Data(:,i);
    s = s + x' * ILambda * x;
end
llh = -nVectors*(dim/2)*log(2*pi) - nVectors*logDetLambda/2 - s/2;
% End of function getEvidence



%=======================================================================
% function  llh = getLikelihood(Data, V, Sigma, Z, IDmat)
%
% Compute log-likelihood of Data based on Eq. 11 of
% PROBABILISTIC METHODS FOR FACE REGISTRATION AND RECOGNITION (Li and Prince)
%
%   Input:
%       Data    - Mean-removed training data (NFeature  x NSample)
%       V       - Factor loading matrix of latent identity variable
%       Sigma   - Covariance matrix of noise (diagonal)
%       Z       - latent variables of speakers
%       IDmat   - ID matrix containing speaker information (N_DATA x N_INDIV)
%
%   Output:
%       llh     - log-likelihood
%=======================================================================
function llh = getLikelihood(Data, V, Sigma, Z, IDmat)
N_INDIV = size(IDmat,2);
L = chol(Sigma);
logDetSig = 2*sum(log(diag(L)));
ISig = inv(Sigma);
llh = 0;
for i = 1:N_INDIV,
    sessionIdx = find(IDmat(:,i));
    nSessions = length(sessionIdx);
    z = Z(:,i);
    for j = 1:nSessions,
        x = Data(:,sessionIdx(j));        
        temp = x - V*z;
        llh = llh - 0.5*(temp'*ISig*temp);
    end
    llh = llh - 0.5*nSessions*logDetSig - 0.5*(z'*z);
end
% End of function getLikelihood

%=======================================================================
% function  [P, Q, const] = prepare_scoring(V,Sigma)
% Compute the matrix P and Q in the paper
% Analysis of I-vector Length Normalization in Speaker Recognition Systems
% Note: To use the scoring function in Garcia-Romero's paper, the global mean
%       of i-vecs must be zero.
%   Input:
%       V      - Speaker factor loading matrix
%       Sigma  - Full residual cov matrix
%   Output:
%       P and Q matrix in the above paper
%       const  - Const. term of log-likelihood
%=======================================================================
function [P, Q, const] = prepare_scoring(V, Sigma)

% \Sigma_tot and \Sigma_ac
Sig_a = V*V';
Sig_t = Sig_a + Sigma;

% {\Sigma_tot}^{-1}
ISig_t = inv(Sig_t);
temp = Sig_t - ((Sig_a / Sig_t) * Sig_a);
Q = ISig_t - inv(temp);
P = (Sig_t \ Sig_a) / temp;

% Compute the constant term
Sig_0 = zeros(size(Sig_a));
D1 = [Sig_t Sig_a; Sig_a Sig_t];
D2 = [Sig_t Sig_0; Sig_0 Sig_t];
const = 0.5*logDet(D2) - 0.5*logDet(D1);
return;

%=======================================================================
% function d = logDet(A)
% Return the log of determinate of a matrix
%=======================================================================
function d = logDet(A)
L = chol(A);
d = 2*sum(log(diag(L)));
