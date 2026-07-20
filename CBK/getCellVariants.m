function [X, sqX, isqX, iX] = getCellVariants(Xin)

    d = size(Xin,1);
    N = size(Xin,3);

    X = cell(N,1);
    for t = 1:N
        X{t} = squeeze(Xin(:,:,t));
    end

    [sqX, isqX, iX] = deal(cell(N,1));

    for t=1:N
        % prevent non-SPD
		X{t} = X{t} + 1e-20*eye(d);
        iX{t} = inv(X{t});
        sqX{t} = sqrtm(X{t});
        isqX{t} = inv(sqX{t});
    end

end