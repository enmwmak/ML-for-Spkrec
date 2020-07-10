function T = comp_Tmatrix(ubm,stats_file, nw, totalvar_mat_file)
% Estimate total variability matrix T with nw total factors
% This function replaces train_t_from_files.m
% Input:
%   ubm                 : UBM file
%   stats_file          : suff stats file
%   nw                  : number of total factors
% Output:
%   totalfac_mat_file   : total factors w of all speakers in the training set
%
% Dehak, et al. (2010) "Front-end factor analysis for speaker verification
% Example:
%    comp_Tmatrix('ubm/fw60/bg.mix.male.cmsY.1024c',{'stats/fw60/male_target-mix_mix_stats_1024c.mat'},500,'mat/fw60/male_mix_t500_1024c.mat');

nIters = 5;
initT = 'Yes';       % 'Yes' if initialize T randomly
fprintf('No. of iterations = %d\n',nIters);
fprintf('Initialize T matrix = %s\n', initT);

% Load suf stats

% we need the ubm params
[m, E, ~] = load_ubm(ubm);
m = m'; E = E';      % Need to be row vectors below

% Load T matrix or init T randomly
if strcmp(initT,'Yes')==1,
    T = init_T_matrix(nw, length(m));
else
    fprintf('Initialize total var matrix from %s\n',totalvar_mat_file);
    load(totalvar_mat_file);
end

% Loading all sufficient stats to memory
fraction = 0.5; F = []; N = []; tot_n_utts = 0;
for j=1:length(stats_file),
    fprintf('Loading suff stats file %s',stats_file{j});
    sstat = load(stats_file{j},'F','N','spk_logical');
    n_utts = round(length(sstat.spk_logical)*fraction);
    fprintf(' (No. of utts = %d)\n', n_utts);
    RP = randperm(length(sstat.spk_logical));
    ridx = RP(1:n_utts);
    F = [F; sstat.F(ridx,:)]; 
    N = [N; sstat.N(ridx,:)];
    clear sstat;
    tot_n_utts = tot_n_utts + n_utts;
end
fprintf('Total no. of utts = %d\n', tot_n_utts);
session_ids = (1:tot_n_utts)';
n_sessions = tot_n_utts;
for i=1:nIters
    fprintf('Iteration %d\n',i);
    [~, T]=estimate_y_and_v_mem(F, N, 0, m, E, 0, T, 0, zeros(tot_n_utts,1), ...
                            0, zeros(n_sessions,1), session_ids);
    disp(['Saving T to ' totalvar_mat_file]); save(totalvar_mat_file, 'T');                          
end


% Pretend that every utt from a given speaker is produced by different speakers
% Note that in the i-vector framework, we do not estimate a subspace
% that describes the between-speaker variability. Instead, the
% directions of greatest between-utterance variability are estimated.
% Therefore, this framework does not require speaker-labelled utterances 
% in the subspace training dataset.
% fraction = 0.5;            % Select a fraction of training data for efficiency 
% for i=1:nIters
%     fprintf('Iteration %d\n',i);
%     for j=1:length(stats_file),
%         fprintf('Using %s to train T',stats_file{j});
%         sstat = load(stats_file{j},'F','N','spk_logical');
%         n_speakers = round(length(sstat.spk_logical)*fraction);
%         fprintf(' (No. of utts = %d)\n', n_speakers);
%         RP = randperm(length(sstat.spk_logical));
%         ridx = RP(1:n_speakers);
%         F = sstat.F(ridx,:); 
%         N = sstat.N(ridx,:);
%         clear sstat;
%         spk_ids = (1:n_speakers)';
%         n_sessions = n_speakers;
%         [~, T]=estimate_y_and_v(F, N, 0, m, E, 0, T, 0, zeros(n_speakers,1), ...
%                                 0, zeros(n_sessions,1), spk_ids);
%         clear F N;                    
%     end
%     disp(['Saving T to ' totalvar_mat_file]); save(totalvar_mat_file, 'T');                          
% end



%% private function
function T = init_T_matrix(iVecDim,gmmsvDim)
% Initialize total variability matrix T randomly
disp('Initializing T matrix (randomly)');
T = randn(iVecDim, gmmsvDim,'double');
for i=1:iVecDim,
    T(i,:) = T(i,:)/norm(T(i,:));
end


