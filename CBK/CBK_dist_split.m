function dist = CBK_dist_split(Xi, Xj, B, alpha, beta, M, k, K, Lindex)

    L = length(Lindex);
    [Vi, Vj] = deal(zeros(k*L,1));

    % Find closest atoms to Xi
    v_pre = zeros(K,1);

    for t=1:K

        Bt = B{t};
        v_pre(t) = ABLD(Xi, Bt, alpha(t), beta(t));

    end

    % Sort to use k 25%, 50%, 75% or 100% of atoms
    [~, minIdx] = sort(v_pre);

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

    % The dichotomy transform
    V = Vi - Vj;

    dist = V'*M*V;

end