function d = geodist(p,x)
%compute geodesic distance between two tensors
%Input: p --- positive definite matrix
%       x --- positive definite matrix
%Output:d --- geodesic distance
[u,L] = eig(p);
g = u*sqrt(L);
if (abs(det(g))==0) 
    d = 10000;
    return;
end
invg = inv(g);
y = invg*x*invg';
[V D] = eig(y);
dd = diag(D);
d = sum(log(dd).^2);
d = sqrt(d);