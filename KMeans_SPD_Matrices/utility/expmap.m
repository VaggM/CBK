function z = expmap(p,X)
%Riemannian Exponential Map
%Input: p --- positive definite matrix
%       X --- symmetric matrix
%Output:z --- positive definite matrix
%---------old version---------------
% [u,L] = eig(p);
% g = u*sqrt(L);
% Y = inv(g)*X*inv(g)';
% [v,S] = eig(Y);
% z = (g*v)*expm(S)*(g*v)';
%--------fast version---------------
[u,L] = eig(p);
g = u*sqrt(L);
invg = inv(g);
Y = invg*X*invg';
[v,S] = eig(Y);
gv = g*v;
z = gv*diag(exp(diag(S)))*gv';