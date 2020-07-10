function Y = gaussian_function(data, a, b, c)
%Y = gaus(data, b, c)
%GAUS N-domensional gaussian function---
%    See http://en.wikipedia.org/wiki/Gaussian_function for definition.
%    note it's not Gaussian distribution as no normalization (g-const) is performed
%
%    Every row data is one input vector. Y is column vector with the
%    same number of rows as data

auxC = -0.5 ./ c;                    % -0.5R   R=inv(Sigma)

aux = bsxfun(@minus, data, b);       % x - mu
aux = aux .^ 2;                      % (x-mu)'(x-mu)
Y = auxC' *aux;                      % -0.5(x-mu)'R(x-mu)  

Y = exp(Y) * a;

