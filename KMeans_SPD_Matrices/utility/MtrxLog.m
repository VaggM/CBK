function res  = MtrxLog(dt)
% Usange : log for sym matrix
% dt: n x n x m, dt = dt'
 
 if (AreAllSym(dt) == 0)%%n ~= n1 || dt' ~= dt
     help Sym2Vct;
     return
 end
 
Temp = (dt+dt')/2;
[V,S] = eig(Temp);
%[V, S, U] = svd(Temp);

D = sqrt(diag(V'*V));
iD = 1./D;
V = V*diag(iD);

res = V*diag(log(diag(S)))*V';