function M = KLS_mean(Xs)

n = size(Xs,3);
dim = size(Xs,1);

A = zeros(dim,dim);
B = zeros(dim,dim);

for i=1:n
    A = A+ Xs(:,:,i);
    B = B+inv(Xs(:,:,i));
end

bsi = sqrtm(inv(B));
bs = sqrtm(B);

M = bsi*sqrtm(bs*A*bs)*bsi;


end