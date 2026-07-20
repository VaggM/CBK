function [m, num] = meanTensor(p)
%Intrinsic Mean of Diffusion Tensors
%Input:  p --- d-by-d-by-n matrix
%Output: m --- d-by-d matrix mean tensor
%initialization
m0 = p(:,:,1);
t = 1;
epsilon = 1e-7;
[d,d,n] = size(p);
X0 = zeros(d,d);
for i=1:n
    X0 = X0 + logmap(m0,p(:,:,i));
end
X0 = X0/n;
X = X0;
m = expmap(m0,t*X);
m0 = m;
num = 0;
while(norm(X,2)>=epsilon && num <1000)
    num = num+1;
    X = zeros(d,d);
    for i=1:n    
        X = X + logmap(m0,p(:,:,i));
    end
    X = X/n;
    m = expmap(m0,t*X);
    if(norm(X,2)>norm(X0,2))
        t = t/2;
        X = X0;
    end
    %disp(num2str(norm(X,2)));
    X0 = X;
    m0 = m;
end