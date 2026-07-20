function Z = logmap(p,x)
%Riemannian Log Map
%Input: p --- positive definite matrix
%       x --- positive definite matrix
%Output:Z --- Symmetric matrix
%----------old version---------------
% [u,L] = eig(p);
% g = u*sqrt(L);
% y = inv(g)*x*inv(g)';
% [v,S] = eig(y);
% Z = (g*v)*logm(S)*(g*v)';
%----------fast version--------------
[u,L] = eig(p);
g = u*sqrt(L);
invg = inv(g);
y = invg*x*invg';
[v,S] = eig(y);
gv = g*v;
Z = gv*diag(log(diag(S)))*gv';