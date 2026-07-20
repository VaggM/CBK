function D = distLogDet(X,Y)

D = log(det((X+Y)/2)) - 0.5*log(det(X*Y));

end