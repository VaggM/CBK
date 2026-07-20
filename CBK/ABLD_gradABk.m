function [grad_ak, grad_bk] = ABLD_gradABk(Ek, lgEk, ak, bk)
    
    Eb = Ek.^(bk); 
    Ema = Ek.^(-ak);
    EE = (ak*Eb + bk*Ema);
    
    CC =  log(ak+bk) - log(EE);
    grad_ak = ((ak*Eb - ak*bk*(Ema.*lgEk))./(EE+eps) -ak/(ak+bk+eps) + CC)/((ak^2)*bk+eps);
    grad_ak = sum(grad_ak, 1);
    % eig of spd matrixes that are inverted are just inverted eig of the
    % original matrix
    grad_bk = ((bk*Ema + ak*bk*(Eb.*lgEk))./(EE+eps) -(bk/(ak+bk+eps)) + CC)/(ak*(bk^2)+eps);
    grad_bk = sum(grad_bk, 1);

end