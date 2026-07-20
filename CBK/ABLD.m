function dist = ABLD(X, B, alpha, beta) 
    % try
        e = eig(X,B);  d=length(e); e(e<=1e-10)=1e-10;
    % catch
    %     disp('error!');
    %     B(isnan(B)) = 0; B = B + 1e-5*eye(size(B,1));
    %     [~,e] = schur(X\B); e=diag(e); d=length(e);
    % end
    dist = (sum(log((alpha*(e.^beta) + beta*(e.^(-alpha))))) - d*log(alpha+beta))/(alpha*beta+eps);
end