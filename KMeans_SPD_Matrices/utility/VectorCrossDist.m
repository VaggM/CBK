function M = VectorCrossDist(V1, V2)

M = zeros(size(V1,3), size(V2, 3));

for i = 1:size(V1, 3)
    for j = 1:size(V2,3)
        M(i,j) = geodist(V1(:,:,i), V2(:,:,j));
    end
end