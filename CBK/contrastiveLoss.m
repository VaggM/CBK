function [obj, dimS, dimD] = contrastiveLoss(V, Y, M, M0, zs, zd, lam)

    N = size(Y,1);

    S = nnz(Y==1);
    D = nnz(Y==0);

    V1 = V(:,1:N);
    V2 = V(:,N+1:end);
    V12 = (V1-V2);

    dth = diag((V12)'*M*(V12));

    similar = Y.*max(0,dth-zs).^2;
    % 1/S only for the distance indicator
    dimS = (1/S)*Y'*dth;

    dissimilar = (1-Y).*max(0,zd-dth).^2;
    % 1/D only for the distance indicator
    dimD = (1/D)*(1-Y)'*dth;

    gammaM = trace(M/M0) - log(det(M/M0)) - length(M);

    obj = (1/S)*sum(similar) + (1/D)*sum(dissimilar) + lam*gammaM;

end