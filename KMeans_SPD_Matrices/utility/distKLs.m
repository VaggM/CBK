function d = distKLs(X1,X2)

d = sqrt(trace(inv(MtrxSqrt(X1))*X2*inv(MtrxSqrt(X1))...
    +inv(MtrxSqrt(X2))*X1*inv(MtrxSqrt(X2)))-2*size(X1,1))/2;

end