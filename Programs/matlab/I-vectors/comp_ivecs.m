function [w, spk_logical] = comp_ivecs(ubm,t_mat_file,stats_file,ivec_file,n_proc)
% Compute the i-vectors of all speakers given T and sufficient statistics
% This function replaces 'comp_target_w.m'
% Input:
%   ubm                 : UBM file
%   t_mat_file          : File containing total factor matrix T
%   stats_file          : File containing sufficient statistics
%   n_proc              : No. of parallel processes
% Output:
%   ivec_file           : File containing i-vectors corresponding to the stats_file
% Example:
%   comp_ivecs('ubm/fw60/bg.mix.male.cmsY.1024c','mat/fw60/male_mix_t500_1024c.mat','stats/fw60/male_target-mix_mix_stats_1024c.mat','mat/fw60/male_target-mix_mix_t500_w_1024c.mat',1);
%
% Dehak, et al. (2010) "Front-end factor analysis for speaker verification

complikelihood = 0;     % Do not compute likelihood to save computataion

if nargin <5,
    error('No. of input arguments should be 5');
end

% Load UBM
[m, E, ~] = load_ubm(ubm);
m = m'; E = E';      % Need to be row vectors below

% Load suf stats
disp 'Loading sufficient statistics';
trn = load(stats_file);
if complikelihood == 0,
    clear trn.S;
end

% Load the factor loading matrices T
disp(['Loading total variability matrix T from ' t_mat_file]);
Tmat = load(t_mat_file); T = Tmat.T; clear Tmat; 

% Estimate total factor w for each speaker
disp('Estimate w for each speaker');
if n_proc > 1,
    if (matlabpool('size') == 0),
        matlabpool(n_proc);
    end
    nf = size(T,1);
    nUtts = size(trn.N,1);
    F = trn.F;
    N = trn.N;
    w = zeros(nUtts,nf);
    parfor i = 1:nUtts,
        fprintf('Estimating i-vector of speaker %d of %d ',i,nUtts);
        w(i,:) = estimate_y_and_v(F(i,:), N(i,:), 0, m, E, 0, T, 0, zeros(1,1),...
                                 0, zeros(1,1), 1);  % one utt at a time
    end
    matlabpool close;
else
    spk_ids = (1:size(trn.N,1))';
    w = estimate_y_and_v(trn.F, trn.N, 0, m, E, 0, T, 0, zeros(size(trn.N,1),1),...
                             0, zeros(size(trn.N,1),1), spk_ids);  
end
    
% Compute likelihood
if complikelihood == 1,
    n_speakers = size(trn.N,1);
    n_sessions = n_speakers;
    llike = comp_loglike(trn.F, trn.N, trn.S, m, E, 0, T, 0, zeros(n_speakers,1), w, zeros(n_sessions,1), spk_ids);
    disp(['Likelihood = ' num2str(llike)]);
end
% Return results
spk_logical = trn.spk_logical;    % Obtained from enroll_stats_file
spk_physical = trn.spk_physical;
num_frames = trn.num_frames;

% Save w to file
disp(['Saving w to ' ivec_file]);
save(ivec_file,'w','spk_logical','spk_physical','num_frames');