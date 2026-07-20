function M = LogEuc_mean(Xs)

n = size(Xs,3);
dim = size(Xs,1);

sum = zeros(dim,dim);

for i=1:n
    sum = sum + MtrxLog(Xs(:,:,i));
end

M = MtrxExp(sum/n);

end
