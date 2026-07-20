function E = ABLD_eigs(X, B, d, n, N)

    E = zeros(d, n, N);
    for t=1:N           
        E(:,:,t) = cell2mat(cellfun(@(b) (eig(X{t}, b)), B, 'uniformoutput', false));            
    end
    E = permute(E,[1,3,2]);

    % safety for zero eigenvalues
    E(E<1e-10)=1e-10;

end