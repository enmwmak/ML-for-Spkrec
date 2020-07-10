function [F,N,spk_logical] = comp_suf_stats(ubm, set_list_file, out_stats_file, n_proc)
% Compute sufficient statistics for estimating i-vec. Based on jfa_cookbook
% Inputs:
%   ubm            : filename of the UBM
%   set_list_file  : A .lst file containing the name of .cep files for computing sufficient stats
%   out_stats_file : Output suf stats file in .mat format 
%   n_proc         : No. of parallel processes
% Outputs:
%   out_stats_file - Sufficient stat files
% Example:
%   comp_suf_stats('ubm/fw60/bg.mix.male.cmsY.1024c','lst/fw60/male_target-mix_stats.lst','stats/fw60/male_target-mix_mix_stats_1024c.mat',1);

if nargin < 3,
    error('No. of input arguments should be 3 or 4');
end
if nargin == 3,
    n_proc = 1;
end

% Load the UBM (or use your own UBM reader)
tic;
[m, v, w] = load_ubm(ubm);

n_mixtures  = size(w, 1);
dim         = size(m, 1) / n_mixtures;

% we load the model as superverctors, so we reshape it to have each gaussian in
% one column
m = reshape(m, dim, n_mixtures);
v = reshape(v, dim, n_mixtures);

% process the dataset
disp(['Processing list ' set_list_file ' to produce ' out_stats_file]);
disp(['No. of mixtures = ' num2str(n_mixtures)]);

% process the file list (logical=physical)
[spk_logical spk_physical] = parse_list(set_list_file);

% Make sure all .cep files exists and have correct feature dim
missing_file = 0;
n_sessions = size(spk_logical, 1);
for i = 1:n_sessions
    session_name = [spk_physical{i,1} '.cep'];
    if exist(session_name,'file')==0,
        fprintf('Error: Line %d of %s. File %s does not exist\n',i,set_list_file,session_name);
        missing_file = missing_file + 1;
    end
    %if (~isFileValid(session_name, dim))
    %    return;
    %end
end
if missing_file > 0,
    return;
end


% initialize the matrices for the stats
% one row per session
N = zeros(n_sessions, n_mixtures,'single');
F = zeros(n_sessions, n_mixtures * dim,'single');
%S = zeros(n_sessions, n_mixtures * dim,'single');
S = [];   % Save some memory. S is only useful if you need to compute likelihood

% process sessions
fprintf('No. of sessions = %d\n', n_sessions);

num_frames = zeros(n_sessions,1);
if n_proc > 1,
    if (matlabpool('size') == 0),
        matlabpool(n_proc);
    end
    parfor i = 1:n_sessions,
        session_name = [spk_physical{i,1} '.cep'];
        data = readcep(session_name);       % Or using your own MFCC reader
        num_frames(i) = size(data,2);
        fprintf('Process %d frames (%d/%d) %s\n',num_frames(i),i,n_sessions, session_name);
        [Ni,Fi,~] = collect_suf_stats(data, m, v, w);
        N(i,:) = Ni';    % N: LxC matrix, where L is no. of utt and C is no. of mix
        F(i,:) = Fi';    % F: LxCD matrix, where D is feature dim
        %S(i,:) = Si';    % S: 2nd-order stats, LxCD
    end
    matlabpool close;
else
    % Create a hash table containing the session index that have been processed
    % Use the session name as the key. <session_name,session_index>
    % Note: Do not concatenate the .lst files in lists/ dir; otherwise, the
    %       hash table may mistakenly consider different speakers of the same
    %       physical name from different corpora to be the same speaker. 
    %       To concatenate the suf stats, use the script concate_suf_stats.m.
    sessionhash = java.util.Hashtable;
    for i = 1:n_sessions
        session_name = [spk_physical{i,1} '.cep'];

        % Put session_name in a hash table so that if the same session name
        % (.cep file) is to be process again in subsequent session, we
        % just copy the sufficient stats that have been computed previously.
        if (sessionhash.containsKey(session_name) == 0),
            sessionhash.put(session_name,i);
            fprintf('\rProcess (%d/%d) %s',i,n_sessions, session_name);
            data = readcep(session_name);       % Or using your own MFCC reader
            num_frames(i) = size(data,2);
            % process the feature file. 
            % Ni = sum_t gamma_t(c), where c=1:C and i corresponds to var s in Kenny08
            % Fi = sum_t gamma_t(c)*x_tj, where j=1:D
            [Ni,Fi,Si] = collect_suf_stats(data, m, v, w);
            N(i,:) = Ni';    % N: LxC matrix, where L is no. of utt and C is no. of mix
            F(i,:) = Fi';    % F: LxCD matrix, where D is feature dim
            %S(i,:) = Si';    % S: 2nd-order stats, LxCD
        else
            session_index = sessionhash.get(session_name);
            fprintf('\nCopy suff stats from session %d to %d for %s\n', session_index,i,session_name);
            num_frames(i) = num_frames(session_index);
            N(i,:) = N(session_index,:);
            F(i,:) = F(session_index,:);
            %S(i,:) = S(session_index,:);
        end
    end
end

% Save the suff stats
disp(['Saving stats to ' out_stats_file]);
save(out_stats_file, 'N', 'F', 'S', 'spk_logical', 'spk_physical', 'num_frames', '-v7.3');
toc;


%% Private function

function v = isFileValid(session_name, dim)
v = 1;
fprintf('%s\n',session_name);
cep = readcep(session_name);
if size(cep,1) ~= dim,
   fprintf('Error: File %s has feature dim %d but UBM has feature dim %d\n',...
       session_name,size(cep,2),dim);
   v = 0;
   return;
end
return;

