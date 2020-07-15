% Posterior of two 1-D Gaussian
% Note: mvnpdf is very slow. We can speedup computation by writing our own function
%       where the det(sigma2) is pre-computed.
%       Another speed up is to input the 2x2 precision matrix
function posty = posterior2G(ls, lt, pi, mu, sigma)
K = length(pi);
wlh = zeros(K,K);
sigmaSq = sigma.^2;
IsigmaSq = 1./sigmaSq;
Isigma = 1./sigma;
for p = 1:K,
    for q = 1:K,
        mu2 = [mu(p); mu(q)];
		detIsigma = Isigma(p) * Isigma(q);
        preMat = diag([IsigmaSq(p) IsigmaSq(q)]);		
        wlh(p,q) = pi(p)*pi(q)*pdf2D([ls; lt], mu2, preMat, detIsigma);
    end
end
temp = sum(sum(wlh));
assert(temp>0,'Divided by zeros in posterior_y2 of mPLDA_GroupScoring.m');
posty = wlh/temp; 

function p = pdf2D(x, mu, R, detIsigma)
temp = x-mu;
R = diag(R);
%p = 0.159154943 * detIsigma * exp(-0.5*temp'*R*temp);
p = 0.159154943 * detIsigma * exp(-0.5*sum(temp.^2.*R));