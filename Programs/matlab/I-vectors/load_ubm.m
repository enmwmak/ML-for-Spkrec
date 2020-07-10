function [m, v, w, ctr, pre, mix] = load_ubm(ubm_basename)
% Load UBM from .ctr, .wth, and .mix files based on ubm_basename
% m: CFx1 mean vector
% v: CFx1 variance (diagnoal of covariance matrix)
% w: CFx1 mixture weights

ctrfile = [ubm_basename '.ctr'];
wthfile = [ubm_basename '.wth'];
mixfile = [ubm_basename '.mix'];

% Read .ctr file
fp = fopen(ctrfile,'rt');
F = fscanf(fp,':%d\n',1);
fscanf(fp,':%d\n',1);           % Skip one line
C = fscanf(fp,':%d\n',1);
ctr = zeros(C,F);
for i=1:C,
    ctr(i,:) = fscanf(fp,'%f ',F);
    fscanf(fp,'%f\n',1);               % Dummy output
end
m = reshape(ctr',C*F,1);
fclose(fp);

% Read .wth file
fp = fopen(wthfile,'rt');
pre = zeros(C,F);
for i=1:C,
    pre(i,:) = fscanf(fp,'%f',F);
end
v = 1./pre;                                % .wth file contains precision matrix
v = reshape(v',C*F,1);
fclose(fp);

% Read .mix file
w = load(mixfile);
mix = w;





