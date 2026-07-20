% =========================================================================
%
%     The Riemannian Conjugate Gradient process is based on the following.
%
%     Discriminative Alpha-Beta Divergence Framework
%     Copyright (c) 2017, Panagiotis Stanitsas and Anoop Cherian.
%     Citation: Cherian et al., Learning Discriminative Alpha-Beta Divergence 
%     for Positive Definite Matrices, ICCV, 2017.
%     Note: Distributed under the BSD 3-Clause License. See root LICENSE file.
%
% =========================================================================


function [B, alpha, beta, M, loss_val, dim_sim, dim_dis, atomIndicator] = CBK(Xi_train, Xj_train, train_labels, B0, a0, b0, M0, params)
    lam = params.lam;
    zs = params.zs;
    zd = params.zd;
    m0 = params.m0;
    
    k = params.k;
    L = params.L;
    Lindex = params.Lindex;

    X_train = cat(3,Xi_train, Xj_train);

    [X, sqX, isqX, ~] = getCellVariants(X_train);

    % sizes
    % N - amount of samples/data
    % d - [dxd] dimentions of each data
    % K - number of IDDL atoms

    N = length(train_labels);
    d = size(X{1},1);
    K = length(B0);
    
    % init abld
    B = B0;
    alpha = a0;
    beta = b0;
    alphabeta = [alpha; beta];
    M = M0;

    [Vsplit, pairIdx, atomIndicator] = updateVsplit(X, B, alpha, beta, k, K, Lindex, N);
    
    % Set Up manifold structures
    % spd manifold powered to the number of atoms
    spd_manifold = powermanifold(sympositivedefinitefactory(d), K);
    spd_problem.M = spd_manifold; 
    spd_problem.cost = @spd_objective;
    spd_problem.egrad = @spd_gradient; 

    % spd manifold for M
    mahal_manifold = sympositivedefinitefactory(k*L);
    mahal_problem.M = mahal_manifold;
    mahal_problem.cost = @mahal_objective;
    mahal_problem.egrad = @mahal_gradient;

    % eucl manifold for vector [a; b]
    manifold = euclideanfactory(2*K, 1);
    alphabeta_problem.M = manifold;
    alphabeta_problem.cost = @alphabeta_objective;
    alphabeta_problem.egrad = @alphabeta_gradient;
    
    % internal common functions
    % vector V is updated every time the objective is calculated
    function obj = objective(B, alpha, beta, M)
        % update V values
        [Vsplit, pairIdx, newIdx] = updateVsplit(X, B, alpha, beta, k, K, Lindex, N);
        atomIndicator = atomIndicator + newIdx;

        obj = contrastiveLoss(Vsplit, train_labels, M, m0, zs, zd, lam);
    end

    % alpha manifold functionals
    function obj = alphabeta_objective(ab)
        obj = objective(B, ab(1:K), ab(K+1:end), M);
    end

    function grad_ab = alphabeta_gradient(ab)

        % prepare eig for Xi||B
        % E is [dxNxK]
        Ei = ABLD_eigs(X(1:N), B, d, K, N);
        lgEi = log(Ei);

        % prepare eig for Xj||B
        % E is [dxNxK]
        Ej = ABLD_eigs(X(N+1:end), B, d, K, N);
        lgEj = log(Ej);

        % current a,b and 
        a = ab(1:K);
        b = ab(K+1:end);

        % dim Nxn
        Y = train_labels;
    
        S = nnz(Y==1);
        D = nnz(Y==0);
    
        V1 = Vsplit(:,1:N);
        V2 = Vsplit(:,N+1:end);
        V12 = (V1-V2);
    
        dth = diag((V12)'*M*(V12));
    
        MM = 2*M; % M + M';
    
        similar = (1/S)*2*Y.*max(0,dth-zs);
    
        dissimilar = (1/D)*2*(Y-1).*max(0,zd-dth);
    
        [grad_a, grad_b] = deal(zeros(K,1));

        [grad_ai, grad_bi, grad_aj, grad_bj] = deal(zeros(K,N));

        % grad for all input data
        % output are vectors [1*N]
        for t=1:K
            [grad_ai(t,:), grad_bi(t,:)] = ABLD_gradABk(Ei(:,:,t), lgEi(:,:,t), a(t), b(t));
            [grad_aj(t,:), grad_bj(t,:)] = ABLD_gradABk(Ej(:,:,t), lgEj(:,:,t), a(t), b(t));
        end
    
        for n=1:N

            x = pairIdx(:,n);
            V12idx = V12(:,n);

            % For a faster demonstration we update the gradient 
            % K times (once per atom) rather than
            % KxL times (L index sets per atom).

            for xidx = 1:length(x)

                t = x(xidx); % Helps when k 25%, 50%, 75% or 100% of atoms
                Midx = MM((xidx-1)*L+1,:);

                mv12 = Midx * V12idx;
    
                grad_a(t) = grad_a(t) + (similar(n)+dissimilar(n))*mv12*(grad_ai(t,n)-grad_aj(t,n));
                grad_b(t) = grad_b(t) + (similar(n)+dissimilar(n))*mv12*(grad_bi(t,n)-grad_bj(t,n));

    
            end
        end

        grad_ab = real([grad_a; grad_b]);
        
        % prevent nan
        grad_ab(isnan(grad_ab)) = 0;
    end

    % spd manifold functionals
    function obj = spd_objective(B) 
        obj = objective(B, alpha, beta, M);
    end

    function gradB = spd_gradient(B)

        Y = train_labels;
    
        S = nnz(Y==1);
        D = nnz(Y==0);
    
        V1 = Vsplit(:,1:N);
        V2 = Vsplit(:,N+1:end);
        V12 = (V1-V2);
    
        dth = diag((V12)'*M*(V12));
    
        MM = 2*M; % M + M';
    
        similar = (1/S)*2*Y.*max(0,dth-zs);
    
        dissimilar = (1/D)*2*(Y-1).*max(0,zd-dth);
    
        gradB = cell(K,1);
        [gradB{:}] = deal(0);

        for n=1:N

            x = pairIdx(:,n);
            V12idx = V12(:,n);

            % For a faster demonstration we update the gradient 
            % K times (once per atom) rather than
            % KxL times (L index sets per atom).

            for xidx=1:length(x)
                
                t = x(xidx); % Helps when k 25%, 50%, 75% or 100% of atoms
                Midx = MM((xidx-1)*L+1,:);

                mv12 = Midx*V12idx;
    
                i = n;
                j = N+n;
    
                Bt = B{t};
                iBt = inv(Bt+eye(d)*1e-5);
    
                gradBit = ABLD_gradBk(isqX{i}, sqX{i}, Bt, iBt, alpha(t), beta(t));
                gradBjt = ABLD_gradBk(isqX{j}, sqX{j}, Bt, iBt, alpha(t), beta(t));
                
                gradB{t} = gradB{t} + real(((similar(n)+dissimilar(n))*mv12)*(gradBit-gradBjt));
    
            end
    
        end

    end

    % mahal manifold functionals
    function obj = mahal_objective(M)
        obj = objective(B, alpha, beta, M);
    end

    function gradM = mahal_gradient(M)

        Y = train_labels;
    
        S = nnz(Y==1);
        D = nnz(Y==0);
    
        V1 = Vsplit(:,1:N);
        V2 = Vsplit(:,N+1:end);
        V12 = (V1-V2);
    
        dth = diag((V12)'*M*(V12));
    
        similar = (1/S)*2*Y.*max(0,dth-zs);
    
        dissimilar = (1/D)*2*(Y-1).*max(0,zd-dth);
    
        gradM = zeros(size(M));
    
        for n=1:N

            v = V12(:,n);

            gradM = gradM + (similar(n)+dissimilar(n))*(v*v');
    
        end

        gradM = real(gradM + lam*(inv(m0)-inv(M)));

    end

    % main iterations.
    opts.maxiter = params.internal_iter;
    opts.verbosity = 0;

    loss_val = zeros(params.iter,1);

    % Block Coordinate Descend
    for iter = 1:params.iter

        B = conjugategradient(spd_problem, B, opts);

        alphabeta = conjugategradient(alphabeta_problem, alphabeta, opts);
        alpha = alphabeta(1:K);
        beta = alphabeta(K+1:end);
        fprintf('\tmean alpha = %0.3f | mean beta = %0.3f\n', mean(alpha), mean(beta))

        if any(isnan([alpha;beta]))
            fprintf('matrices are not conditioned properly... nans in alpha, beta\n');
            break;
        end

        %% Update Mahalanobis Matrix

        M = conjugategradient(mahal_problem, M, opts);

        %% Print new accuracy

        [loss_val(iter), dim_sim, dim_dis] = contrastiveLoss(Vsplit, train_labels, M, m0, zs, zd, lam);
        fprintf('\titer = %d : Loss = %0.2f\n', iter, loss_val(iter))

    end
end
