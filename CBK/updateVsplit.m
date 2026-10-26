function [V, pairAtomIdx, atomIndicator] = updateVsplit(X, B, alpha, beta, k, K, Lindex, N)

    L = length(Lindex);
    V = zeros(k*L,2*N);
    pairAtomIdx = zeros(k,N);
    atomIndicator = zeros(K,1);

    for n=1:N

        Xi = X{n};
        Xj = X{N+n};

        % Find closest atoms to Xi to use k 25%, 50%, 75% or 100% of atoms

        if k~=K
            v_pre = zeros(K,1);
    
            for t=1:K
    
                Bt = B{t};
                v_pre(t) = ABLD(Xi, Bt, alpha(t), beta(t));
    
            end

            [~, minIdx] = sort(v_pre);

        else

            minIdx = 1:K;

        end
        
        atomIndicator(minIdx(1:k)) = atomIndicator(minIdx(1:k)) + 1;
        pairAtomIdx(:,n) = minIdx(1:k);

        [Vi, Vj] = deal(zeros(k*L,1));

        for t=1:k

            idx = minIdx(t);

            Bt = B{idx};
            
            for l=1:L
        
                lidx = Lindex{l};
    
                vidx = (t-1)*length(Lindex) + l;
    
                Vi(vidx) = ABLD(Xi(lidx,lidx), Bt(lidx,lidx), alpha(idx), beta(idx));
                Vj(vidx) = ABLD(Xj(lidx,lidx), Bt(lidx,lidx), alpha(idx), beta(idx));
    
            end
            
        end

        V(:,n) = Vi;
        V(:,N+n) = Vj;

    end
end
