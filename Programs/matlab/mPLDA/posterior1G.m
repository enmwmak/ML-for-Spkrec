% Posterior of one 1-D Gaussian
% Note: mvnpdf is very slow. We can speedup computation by writing our own function
%       where the det(sigma2) is pre-computed.
%       Another speed up is to input the 2x2 precision matrix
function posty = posterior1G(l, pi, mu, sigma)
K = length(pi);
posty = ones(K,1);
wlh = zeros(1,K);
for r = 1:K,
    wlh(r) = pi(r)*pdf1D(l,mu(r),1/(sigma(r)^2),1/sigma(r));
end
temp = sum(wlh);
assert(~isnan(temp),'Assertion in posteror_y1: sum(wlh) is NaN');
for k = 1:K,
    posty(k) = wlh(k)/temp;
end

function p = pdf1D(x, mu, R, detIsigma)
temp = x-mu;		
p = 0.39894228040 * detIsigma * exp(-0.5*temp*R*temp);