%==============================================================================
% function mplda = mPLDA_train4(X, L, IDmat, nIters, M, K)
% Implement snr-dependent mixture of Gaussian PLDA (training).
% The mPLDA model has K loading matrices and K covmat. 
% This function assumes that each speaker share the same z_i and that z_i depends
% on y_ijk for all j in H_ik and for all k. These assumptions lead to the sum over
% all i-vecs aligned to mixture k when computing <z_i|y_i..,X>. The full set of 
% equations are given in Eq. 39 and 40 of the supplmentary materials.
% This function uses the 2-stage EM as described in Tipping and Bishop's paper.
%
% Input:
%   X         - D x T data matrix with D-dim column vectors
%   L         - T-dim array containing the SNR or the number of frames in the utts
%   IDmat     - ID matrix (numVecs x numSpks). For each row, the column with a 1
%               means that the i-vector is produced by the corresponding speaker.
%               The rest should be zero.
%   M         - No. of speaker factors (latent variables)
%   K         - No. of factor analyzers
%   nIters    - No. of EM iterations
%
% Output:
%   mplda     - A structure containing the following fields
%      pi        - Mixture weights
%      mu        - K x 1 vector containing the mean of SNR
%      sigma     - K x 1 vector containing the stddev of utt length
%      m         - Cell array containing K cells, each contains a (D x 1) mean vectors
%      V         - Cell array containing K cells, each contains a (D x M) factor loading matrices
%      Sigma     - Cell array containing K cells, each contains a (D x D) full covariance matrix of noise e
%      const     - const for computing log-likelihood during scoring 
%      logDetCov - Output of private function prepare_scoring() for fast scoring
%      Icov      - Output of private function prepare_scoring() for fast scoring
%      logDetCov2- Output of private function prepare_scoring() for fast scoring
%      Ico2      - Output of private function prepare_scoring() for fast scoring
%      P,Q       - Cell array of P and Q matrices for fast scoring (used by my APSIPA'15 paper)
%
% Bug fix:
%   2/8/15       - sum_ij V{k}<y_ijk z_ik|X> is not zero. So, it is necessary to subtract it
%                  from sum_ij <y_ijk|X>x_ij when computing m{k}. However, this fix leads to
%                  poor performance. According to Tipping and Bishop paper "Mixtures of 
%                  Probabilistic Principal Component Analysers, if we use the so-called 2-step EM,
%                  we do not need to subtract V{k}<y_ijk z_ik|X> from sum_ij <y_ijk|X>x_ij, i.e.,
%                  m{k} = sum3/sum4;
%
% By M.W. Mak, July 2015
%========================================================================
function mplda = mPLDA_train4(X, L, IDmat, nIters, M, K)
D = size(X,1);              % Dim of i-vectors
N = size(IDmat,2);          % No. of speakers

% Initialize mPLDA model parameters
[pi,mu,sigma,m,V,Sigma] = init_mPLDA(X, L, IDmat, M, K);

% Prepare EM: Create storage for posterior expectations
% To reduce memory consumption, we do not store E_yz and E_yzz. These 
% posterior expectation should be computed on the fly.
% Instead, we store E_z{i} and E_zz{i}. They are independent of j and k
E_y = cell(N,K);                % Hi-dim vector: E_y{i,k}(j) = <y_ijk|L>
E_z = cell(1,K);                % Cell array of M-dim vector E_z{i} = <z_i|y_i..,X>
E_zz = cell(1,K);               % Cell array of MxM matrix E_zz{i} = <z_i z_i|y_i..,X>
H = zeros(N,1);                 % H(i) = No. of sessions for speaker i
sessions = cell(N,1);           % Contains sessions indexes of each training speaker

% Find the sessions of each training speaker
for i = 1:N,
    sessions{i} = find(IDmat(:,i)==1);      % Sessions of spk i
    H(i) = length(sessions{i});
end

% Start mPLDA EM
for t = 1:nIters,   

    % Stage 1: Optimize mixture parameters pi{k},sigma{k}, mu{k}, and m{k}
    [~, posty] = get_alignment(L, sessions, pi, mu, sigma);  % Find the mixture alignment of each utterance
    [pi,mu,sigma,m] = update_mixture(pi,mu,sigma,m,L,X,sessions,posty);
    
    % Stage 2: Use updated mixture parameters to perform EM to optimize V{k} and Sigma{k}
    % E-step: Compute posterior expectation
    [~, posty] = get_alignment(L, sessions, pi, mu, sigma);  % Find the mixture alignment of each utterance    
    for i = 1:N,                                        % For each spk
        ss = sessions{i};                               % Sessions of spk i
        sum_k = zeros(M,M);
        for k = 1:K,
            E_y{i,k} = posty{i}(:,k);
            for j = 1:H(i),
                sum_k = sum_k + posty{i}(j,k)*(V{k}'/Sigma{k})*V{k};
            end
        end
        IL = eye(M)/(eye(M) + sum_k);               % inv(L_i)
        sum_kj = zeros(M,1);
        for k = 1:K,
            for j = 1:H(i),
                x = X(:,ss(j));
                sum_kj = sum_kj + posty{i}(j,k)*(V{k}'/Sigma{k})*(x - m{k});
            end
        end
        E_z{i} = IL*sum_kj;                            % <z_i|y_i..,X>
        E_zz{i} = IL + E_z{i}*E_z{i}';                 % <z_i|y_i..,X><z_i|y_i..,X>'
    end
    
    % M-step  
    % Update V_k
    for k = 1:K,
        sum1 = zeros(D,M);
        sum2 = zeros(M,M);
        for i = 1:N,
            ss = sessions{i};
            for j = 1:length(ss),
                x = X(:,ss(j));
                E_yijk = E_y{i,k}(j);
                sum1 = sum1 + (x-m{k})*E_yijk*E_z{i}';  
                sum2 = sum2 + E_yijk*E_zz{i};
            end
        end
        V{k} = sum1/sum2;       %sum1*inv(sum2);
        %figure; imagesc(V{k}); title(sprintf('V{%d}',k));
    end
    clear sum1 sum2;
    
    % Update tied Sigma_k, equal for all k
    Sigma = update_Sigma(X, m, V, E_y, E_z, sessions);

    % Show some statistics about <y_ijk|L>
    show_mixture_stat(pi, mu, sigma);
    
    % Compute Q-func
    Qnew = auxFunc(pi,mu,sigma,m,V,Sigma, X, L, E_y, E_z, E_zz, sessions);
    if t > 1,
        perinc = abs(100*(Qnew - Qold)/Qold);
        if Qnew > Qold,
            fprintf('Iteration %d: llh = %.2f; %%increase = %.2f%%\n',t,Qnew,perinc);
        else
            fprintf('Iteration %d: llh = %.2f; %%decrease = %.2f%%\n',t,Qnew,perinc);
        end
    end
    Qold = Qnew;
end

% Prepare paras for speeding up scoring
[P,Q, const, Icov, logDetCov, Icov2, logDetCov2] = prepare_scoring(m, V, Sigma, 5);

% Return mPLDA model
mplda.pi = pi;
mplda.mu = mu;
mplda.sigma = sigma;
mplda.m = m;
mplda.V = V;
mplda.Sigma = Sigma;
mplda.const = const;
mplda.logDetCov = logDetCov;
mplda.Icov = Icov;
mplda.logDetCov2 = logDetCov2;
mplda.Icov2 = Icov2;
mplda.P = P;
mplda.Q = Q;
return;

%=======================================================================
% function Sigma = update_tied_Sigma(X, m, V, E_y, E_z, sessions)
%
% Update tied covariance matrices, i.e., Sigma{k} are identical
%
%   Input:
%       X           - D x T data matrix with D-dim column vectors
%       sessions    - Cell array containing the session indexes (index to L) of N speakers
%       m           - Kx1 cell array containing the mean i-vectors of mixtures
%       V           - Cell array of loading matrices
%       E_y         - Cell array of H_i dim vectors E_y{i,k}(j) = <y_ijk|L>
%       E_z         - Cell array of M-dim vector: E_z{i} = <z_i|y_i..,X,L>
%   Output:
%       Sigma       - Cell array of covariance matrices
%=======================================================================
function Sigma = update_tied_Sigma(X, m, V, E_y, E_z, sessions)
K = length(m);
Sigma = cell(K,1);
N = length(sessions);
D = size(X,1);
sum1 = zeros(D,D);
sum2 = 0;
for k = 1:K,
    for i = 1:N,
        ss = sessions{i};
        for j = 1:length(ss),
            x = X(:,ss(j));
            E_yijk = E_y{i,k}(j);
            temp = x - m{k};
            sum1 = sum1 + E_yijk*(temp*temp') - V{k}*E_yijk*E_z{i}*temp';
            sum2 = sum2 + E_yijk;
        end
    end
end
for k = 1:K,
    Sigma{k} = sum1/sum2;
end

%=======================================================================
% function Sigma = update_Sigma(X, m, V, E_y, E_z, sessions)
%
% Update non-tied covariance matrices, i.e., Sigma{k} are not identical
%
%   Input:
%       X           - D x T data matrix with D-dim column vectors
%       sessions    - Cell array containing the session indexes (index to L) of N speakers
%       m           - Kx1 cell array containing the mean i-vectors of mixtures
%       V           - Cell array of loading matrices
%       E_y         - Cell array of H_i dim vectors E_y{i,k}(j) = <y_ijk|L>
%       E_z         - Cell array of M-dim vector: E_z{i} = <z_i|y_i..,X,L>
%   Output:
%       Sigma       - Cell array of covariance matrices
%=======================================================================
function Sigma = update_Sigma(X, m, V, E_y, E_z, sessions)
K = length(m);
Sigma = cell(K,1);
N = length(sessions);
D = size(X,1);
for k = 1:K,
    sum1 = zeros(D,D);
    sum2 = 0;
    for i = 1:N,
        ss = sessions{i};
        for j = 1:length(ss),
            x = X(:,ss(j));
            E_yijk = E_y{i,k}(j);
            temp = x - m{k};
            sum1 = sum1 + E_yijk*(temp*temp') - V{k}*E_yijk*E_z{i}*temp';
            sum2 = sum2 + E_yijk;
        end
    end
    Sigma{k} = sum1/sum2;
    %figure; plot(Sigma{k}); title(sprintf('Sigma{%d}',k));
    %issym = @(x) isequal(tril(x), triu(x)');
    %assert(issym(Sigma{k})==1,sprintf('Sigma{%d} not symmetric',k));        
end            

%=======================================================================
% Display the mixture statistics
%=======================================================================
function show_mixture_stat(pi, mu, sigma)
for k = 1:length(pi),
    fprintf('(%.1f,%.1f,%.1f) ',pi(k),mu(k),sigma(k));
end
fprintf('\n');

%=======================================================================
% function [pi,mu,sigma,m] = update_mixture(pi,mu,sigma,m,L,X,sessions,posty)
%
% Update mixture parameters: pi, mu, sigma, and m
%
%   Input:
%       X           - D x T data matrix with D-dim column vectors
%       L           - Array containing SNR or utterance length of T utterances
%       sessions    - Cell array containing the session indexes (index to L) of N speakers
%       pi,mu,sigma - 1-D GMM of length or SNR model
%       m           - Kx1 cell array containing the mean i-vectors of mixtures
%       posty       - Nx1 cell array, where postp{i} is a H_i x K matrix containing the posterior
%                     probability of K mixtures for H_i sessions.
%   Output:
%       pi,mu,sigma,m
%=======================================================================
function [pi,mu,sigma,m] = update_mixture(pi,mu,sigma,m,L,X,sessions,posty)
K = length(pi);
sum_Ey = zeros(1,K);
D = size(X,1);
N = length(sessions);
for k = 1:K,
    sum1 = 0; sum2 = 0; sum3 = zeros(D,1); sum4 = 0;
    for i = 1:N,
        ss = sessions{i};                           % Sessions of spk i
        for j = 1:length(ss),
            E_yijk = posty{i}(j,k);
            sum1 = sum1 + E_yijk*L(ss(j));
            sum2 = sum2 + E_yijk*(L(ss(j))-mu(k))^2;
            sum3 = sum3 + E_yijk*X(:,ss(j));
            sum4 = sum4 + E_yijk;
        end
    end
    mu(k) = sum1/sum4;
    sigma(k) = sqrt(sum2/sum4);
    m{k} = sum3/sum4;
    sum_Ey(k) = sum4;
end
pi = sum_Ey/sum(sum_Ey);

%=======================================================================
% function [H_set, postp] = get_alignment(L, sessions, pi, mu, sigma)
%
% Find the mixture alignment of each utterance
%
%   Input:
%       L           - Array containing SNR or utterance length of T utterances
%       sessions    - Cell array containing the session indexes (index to L) of N speakers
%       pi,mu,sigma - 1-D GMM of length or SNR model
%   Output:
%       H_set       - NxK cell matrix, where H_set{i,k} is an array containing mixture indexes
%                     of H_i alignments
%       postp       - Nx1 cell array, where postp{i} is a H_i x K matrix containing the posterior
%                     probability of K mixtures for H_i sessions.
%=======================================================================
function [H_set, postp] = get_alignment(L, sessions, pi, mu, sigma)
N = length(sessions);
K = length(pi);
H_set = cell(N,K);
postp = cell(N,1);
for i = 1:N,                                        % For each spk
    ss = sessions{i};                               % Sessions of spk i
    postp{i} = posterior_y(L(ss), pi, mu, sigma);   % <y_ikj|L> where j=1,...,H_i
    [~,maxk] = max(postp{i},[],2);                  % Find the aligned mixture
    for k = 1:K,
        H_set{i,k} = find(maxk == k); 
    end
end

%=======================================================================
% function  [princComp meanVec] = trainPCA(data)
%
% Principla Component Analysis (PCA)
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
% function  Q = auxFunc(pi,mu,sigma,m,V,Sigma, X, L, E_y, E_z, E_zz,, sessions);  
%
%   Input:
%     pi        - Mixture weights
%     mu        - K x 1 vector containing the mean of utt length
%     sigma     - K x 1 vector containing the stddev of utt length
%     m         - Cell array containing nMix (D x 1) mean vectors
%     V         - Cell array containing K (D x M) factor loading matrices
%     Sigma     - (D x D) full covariance matrix of noise e
%     X         - Training data (NFeature  x NSample)
%     L         - Length of utterance, 1 x NSample
%     E_y       - E_y{i,k}(j) = <y_ijk|L>
%     E_z       - E_z{i} = <z_i|y_i..,X,L>
%     E_zz      - Cell array of MxM matrix E_zz{i} = <z_i z_i|y_i..,X>
%     sessions  - Cell array containing no. of sessions for each speaker
%
%   Output:
%       Q       - complete data log-likelihood
%=======================================================================
function  Q = auxFunc(pi,mu,sigma,m,V,Sigma, X, L, E_y, E_z, E_zz, sessions)
N = length(sessions);               % No. of speakers
K = length(pi);
logDetSig = zeros(1,K);             % log|Sigma_k|
ISig = cell(1,K);                   % inv(Sigma_k)
for k = 1:K,
    logDetSig(k) = 2*sum(log(diag(chol(Sigma{k})))); % log|Sigma|
    ISig{k} = inv(Sigma{k});
end
sum1 = zeros(1,K); 
sum2 = zeros(1,K); 
sum3 = zeros(1,K); 
sum4 = zeros(1,K);
sum_ij = zeros(1,K);
for i=1:N,
    ss = sessions{i};   % Sessions of spk i
    for k = 1:K,
        for j = 1:length(ss),
            E_yijk = E_y{i,k}(j);
            x = X(:,ss(j));
            temp = x-m{k};
            sum1(k) = sum1(k) + E_yijk*(-log(sigma(k))-(0.5/(sigma(k)^2))*(L(ss(j))-mu(k))^2);
            sum2(k) = sum2(k) + E_yijk*(log(pi(k))-0.5*logDetSig(k)-0.5*temp'*(ISig{k}*temp));
            sum3(k) = sum3(k) + (temp'*ISig{k})*(V{k}*E_yijk*E_z{i});
            sum4(k) = sum4(k) - 0.5*trace((V{k}'*(ISig{k}*V{k})+1)*(E_yijk*E_zz{i}));
            sum_ij(k) = sum_ij(k) + E_yijk;
        end
    end
end
S = sum1+sum2+sum3+sum4;
fprintf('Likelihood: ');
for k=1:K,
    fprintf('(%.2f,%.2f) ',S(k)/size(X,2),sum_ij(k)/size(X,2));
end
fprintf('\n');
Q = sum(S);
return;

%=======================================================================
% function [P,Q,const, Icov, logDetCov, Icov2, logDetCov2] = prepare_scoring(m, V, Sigma, alpha) 
% Precompute matrices for fast scoring
%   Input:
%       m           - Cell array containing means of i-vectors
%       V           - Cell array containing K DxM Speaker factor loading matrix
%       Sigma       - Cell array containing K DxD full residual cov matrices
%       alpha       - Scaling factor for computing logDetCov, i.e., |alpha*C|, 
%                     to avoid overflow and underflow
%   Output:
%       const       - Const. term of log-likelihood
%       P,Q         - Cell array containing K matrices
%       Icov        - Cell array containing Inverse of V*V'+Sigma{k}
%       logDetCov   - Cell array containing logDet of alpha*cov matrices
%       Icov2       - Same as Icov but based on the stacking of two V
%       logDetCov2  - Same as logDetCov but based on the stacking of two V
% Bug fixed (Jun 15): 
%       Sigma2 should be computed inside ks and kt double-loop
%       Original code: 
%         for ks=1:K,
%             Icov{ks} = inv(V{ks}*V{ks}' + Sigma{ks});
%             logDetCov{ks} = logDet(alpha*(V{ks}*V{ks}' + Sigma{ks}));
%             Sigma2{ks} = [Sigma{ks} zeros(D,D); zeros(D,D) Sigma{ks}]; <--- Bug here, see below fix
%             for kt=1:K,
%                 V2 = [V{ks}; V{kt}];
%                 Icov2{ks,kt} = inv(V2*V2' + Sigma2{ks});
%                 logDetCov2{ks,kt} = logDet(alpha*(V2*V2' + Sigma2{ks}));
%             end
%         end
%        Effect of fixing this bug: Increase EER but decrease minDCF
%=======================================================================
function [P, Q, const, Icov, logDetCov, Icov2, logDetCov2] = prepare_scoring(m, V, Sigma, alpha)
D = size(Sigma{1},1);
K = length(m);
logDetCov = cell(1,K);
Icov = cell(1,K);
logDetCov2 = cell(K,K);
Icov2 = cell(K,K);
for ks=1:K,
    Icov{ks} = inv(V{ks}*V{ks}' + Sigma{ks});
    logDetCov{ks} = logDet(alpha*(V{ks}*V{ks}' + Sigma{ks}));
    for kt=1:K,
        V2 = [V{ks}; V{kt}];
        %Sigma2 = [Sigma{ks} zeros(D,D); zeros(D,D) Sigma{ks}];  % Old
        Sigma2 = [Sigma{ks} zeros(D,D); zeros(D,D) Sigma{kt}];   % Bug fixed 
        Icov2{ks,kt} = inv(V2*V2' + Sigma2);
        logDetCov2{ks,kt} = logDet(alpha*(V2*V2' + Sigma2));
    end
end

% Compute matrices P and Q  and log-likelihood constant for each mixture
P = cell(K,K);
Q = cell(K,K);
Phi = cell(1,K);
Psi = cell(K,K);
const = zeros(K,K);
for i = 1:K,
    Phi{i} = V{i}*V{i}' + Sigma{i};
    for j = 1:K,
        Psi{i,j} = V{i}*V{j}';
    end
end
for ks = 1:K,
    for kt = 1:K,
        Q{ks,kt} = inv(Phi{ks}) - inv(Phi{ks} - ((Psi{ks,kt}/Phi{kt})*Psi{kt,ks}));
        P{ks,kt} = (Phi{ks} \ Psi{ks,kt}) / (Phi{kt} - ((Psi{kt,ks}/Phi{ks})*Psi{ks,kt}));
    end
end
for ks = 1:K,
    for kt = 1:K,
        Sig_0 = zeros(size(Phi{ks}));
        D1 = [Phi{ks} Psi{ks,kt}; Psi{kt,ks} Phi{kt}];
        D2 = [Phi{ks} Sig_0; Sig_0 Phi{kt}];
        const(ks,kt) = -0.5*logDet(D1) + 0.5*logDet(D2) ...
                       - 0.5*m{ks}'*Q{ks,kt}*m{ks} - 0.5*m{kt}'*Q{kt,ks}*m{kt} ...      % Bug fixed. Should be 0.5*
                       - 0.5*m{ks}'*P{ks,kt}*m{kt} - 0.5*m{kt}'*P{ks,kt}'*m{ks};
    end
end

return;

%=======================================================================
% function mplda = init_mPLDA(X, L, IDmat, M, K)
% Initialize the mPLDA model parameters
% Input:
%   X     - D x H matrix containing i-vectors of the same speaker in columns
%   L     - 1 x H vector containing length of utterances
%   IDmat - ID matrix containing speaker information (N_DATA x N_INDIV)
%   M     - Dim of latent vectors
%   K     - No. of mixtures
% Output:
%       pi    - Mixture weights
%       mu    - K x 1 vector containing the mean of utt length
%       sigma - K x 1 vector containing the stddev of utt length
%       m     - Cell array containing K (D x 1) mean vectors
%       V     - Cell array of (D x M) factor loading matrices
%       Sigma - (D x D) full covariance matrix of noise e
%=======================================================================
function [pi,mu,sigma,m,V,Sigma] = init_mPLDA(X, L, IDmat, M, K)
[D,T] = size(X);
Data = double(X) - repmat(mean(X,2), 1, T);
clusterMeans = (Data * IDmat) * (diag(1 ./ (sum(IDmat))));
evec =  trainPCA(clusterMeans);
V = cell(1,K);
for k=1:K,
    V{k} = evec(:,1:M);
end
clusterMeans = (X * IDmat) * (diag(1 ./ (sum(IDmat))));
[~,cmeans] = kmeans(clusterMeans',K);
m = cell(1,K);
for k=1:K,
    m{k} = cmeans(k,:)';
end
Sigma = cell(1,K);
for k = 1:K,
    Sigma{k} = 0.01*eye(D);
end
pi = zeros(K,1);
mu = zeros(K,1);
sigma = zeros(K,1);
[cidx,~] = kmeans(L,K);     % Should not use L' for Matlab2016. But L' works for Matlab2009
for k = 1:K,
    pi(k) = length(find(cidx==k))/length(cidx);
    mu(k) = mean(L(cidx==k));
    sigma(k) = std(L(cidx==k));
end






