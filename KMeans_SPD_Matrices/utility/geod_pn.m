function M = geod_pn(X1, X2, w)

 %sX1 = sqrtm(X1);
 sX1 = chol(X1)';
 isX1 = inv(sX1);
 V = MtrxLog(isX1*X2*isX1')*w;
 M = sX1*MtrxExp(V)*sX1';
 