function bfg = AreAllSym(dt)
% test whether all matrices in the group are symmetric
% Usange :dt: n x n x m,
[n, n1, m] = size(dt);
 if (n ~= n1 )%|| t' ~= dt
     bfg = 0;
     return
 end
%  for i = 1:m
%     t = dt(:,:,i);
%     if t ~= t'
%         bfg = 0;
%         return;
%     end
%  end
 bfg = 1;