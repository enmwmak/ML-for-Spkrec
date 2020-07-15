%=======================================================================
% function d = logDet(A)
% Return the log of determinate of a matrix
%=======================================================================
function d = logDet(A)
L = chol(A);
d = 2*sum(log(diag(L)));
