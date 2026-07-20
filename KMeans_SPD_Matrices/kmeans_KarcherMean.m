function [A, gIdx, flag] = kmeans_KarcherMean(X,k)
flag = 1;
%% Description
% K-Means for a collection of n SPD matrices
%
% Inputs:
%    X - n SPD matrices (dim-by-dim-by-n)
%    k - the number of clusters
% 
% Outputs:
%    A - k SPD cluster centers (dim-by-dim-by-k)
%    gIdx - Cluster indices assigned to the input tensors (n-by-1)
% 
% Dissimilarity Measure:
%    d(X,Y): geodesic distance between X and Y on P(n) manifold
%
% Mean Computation Algorithm:
%    Batch-mode Karcher Mean [1]
%
%% Citation:
%    If you use this software please cite the following papers:
%        [1] Guang Cheng, Hesamoddin Salehian, Baba C. Vemuri “Efficient
%            Recursive Algorithms for Computing the Mean Diffusion Tensor
%            and Applications to DTI Segmentation ”, European Conference on
%            Computer Vision (ECCV) 2012.
%        [2] Jeffrey Ho, Guang Cheng, Hesamoddin Salehian, Baba C. Vemuri, 
%            “Recursive Karcher Expectation Estimators And Geometric Law of
%            Large Numbers”, International Conference on Artificial
%            Intelligence and Statistics (AISTATS) 2013.
%        [3] Hesamoddin Salehian, Guang Cheng, Baba C. Vemuri, Jeffrey Ho,
%            “Recursive Estimation of the Stein Center of SPD Matrices
%            & its Applications”, International Conference on
%            Computer Vision (ICCV) 2013.
%
%
%% Initialization
[dim,dim,n] = size(X);

% randomly select k SPD matrices from X as the initial A
A = X(:,:,1:k);

% allocate variables
g0 = ones(n,1);
gIdx = zeros(n,1);
D = zeros(n,k);

% Total distance computation time
T_dist = 0;
% Total mean computation time
T_mean = 0;

%% Main loop
iter = 1;

while any(g0~=gIdx)
    % disp(['iteration:',num2str(iter),' , ',num2str(sum(g0~=gIdx)),' traveling matrices']);
    g0 = gIdx;
    
    % compute the distance matrix D
    % disp('Computing distances ...');
    tic    
    for i=1:n
        for j=1:k
            D(i,j) = geodist(X(:,:,i),A(:,:,j));
        end
    end
    T_dist = T_dist + toc;
    % disp('Distance computation finished!');
    
    % assign each data to the closest centroid
    [z,gIdx]=min(D,[],2);
    
    % update centroids using means of partitions
    % disp('Updating the cluster centers ...');
    tic    
    for i=1:k
        idx = (gIdx==i);
        if (sum(idx)==0)
            A(:,:,i) = X(:,:,ceil(rand(1,1)*n));
        else
            [A(:,:,i) NN] = meanTensor(X(:,:,gIdx==i));
        end
    end
    T_mean = T_mean + toc;    
    % disp('Cluster centers updated!');
    iter = iter+1;
    if iter==50
        flag = 0; 
        break;
    end
end

% % % disp(['Total iterations: ', num2str(iter)]);
% % % disp(['Total distance computation time (s): ', num2str(T_dist)]);
% % % disp(['Total mean computation time (s): ', num2str(T_mean)]);