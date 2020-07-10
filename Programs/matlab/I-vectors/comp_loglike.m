function llike = comp_loglike(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
% Compute the total log-likelihood

svdim = size(v,2);
spk_llike = zeros(length(unique(spk_ids)),1);    % Log-likelihood for each speaker
for ii = unique(spk_ids)'
    speakers_sessions = find(spk_ids == ii);
    Os = y(ii,:)*v + (z(ii,:).*d).*ones(1,svdim);              
    Oh = zeros(length(speakers_sessions),svdim);    % O_h(s) in Lenny's paper
    h = 1;
    for jj = speakers_sessions'
        Oh(h,:) = Os + (x(jj,:)*u).*ones(1,svdim);
        h = h+1;
    end
    Fh = F(speakers_sessions,:);
    Sh = S(speakers_sessions,:);
    Nh = N(speakers_sessions,:);
    spk_llike(ii) = comp_spkloglike(Fh, Nh, Sh, Oh, E);
%    fprintf('Spk %d: %f\n',ii,spk_llike(ii));
end
llike = mean(spk_llike);



%% Private function
function spk_llike = comp_spkloglike(Fh, Nh, Sh, Oh, E)
% Return the speaker loglikelihood as defined in Kenny's paper

n_sessions = size(Fh,1);
n_mix = size(Nh,2);
dim = size(Fh,2)/n_mix;
t0 = 0;    % Term 0: sum(N_hc*log(1/(2pi)^F/2|Sigma_c|^0.5)
t1 = 0;    % Term 1: -0.5*tr(RS_h)
t2 = 0;    % Term 2: O_h R F_h where R is inverse of cov
t3 = 0;    % Term 3: -0.5*O_h N_h R O_h
Sigma = reshape(E, dim, n_mix);
const = (2*pi)^(dim/2);
for i=1:n_sessions,
    for c=1:n_mix,
        t0 = t0 + Nh(i,c)*log(1/(const*prod(Sigma(:,c))^0.5)); 
    end
    t1 = t1 - 0.5*sum(Sh(i,:)./E);
    vec = (Fh(i,:)./E)';         
    t2 = t2 + Oh(i,:)*vec;
    vec = repmat(Nh(i,:),1,dim).*(Oh(i,:)./E);
    t3 = t3 - 0.5*Oh(i,:)*vec';
end
num_frms = sum(sum(Nh,2));
spk_llike = (t0+t1+t2+t3)/num_frms;

%fprintf('t0=%f; t1=%f; t2=%f; t3=%f; llike=%f\n',t0,t1,t2,t3,spk_llike);
 
