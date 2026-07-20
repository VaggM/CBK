function [paris] = gen_pairs(label,batchsize,rat_gen,rat_rf,rat_sf,Gstop_idx)
%%which nosigs~=nosigsf as it is reflected in the label parameter
% label: the labels of the samples (i.e 1:G1-1, 2:G1-2, ...3:G2-1,... 1:SF1-1, 2:SF2-1, ...)
% rat_gen: how many gens on the batch
% rat_rf: how many RF's on the batch
% rat_sf: how many SF's on the batch
% batchsize=(rat_gen+rat_rf+rat_sf)*batch_mult, for code purposes. 
% Gstop_idx: Index that indicates the end of the Genuine samples, i.e.
% (Gstop_idx+1) indicates the 1st skilled forgery sample, etc. 

num=0;
ratio=rat_gen+rat_rf+rat_sf;
for i = 1:(batchsize/(ratio))
    for j = 1 : rat_gen
        sim_1=unidrnd(Gstop_idx);
        while(1)
            sim_2=unidrnd(Gstop_idx);
            if label(sim_1)==label(sim_2) && sim_1~=sim_2 % G-G
                break;
            end
        end
        paris(num+1,1)=sim_1;
        paris(num+1,2)=sim_2;
        paris(num+1,3)=1;
        num=num+1;
    end
    %% for random forgeries
    for j = 1 : rat_rf
        dis_1=unidrnd(Gstop_idx);
        while(1)
            dis_2=unidrnd(Gstop_idx);
            if label(dis_1)~=label(dis_2) % G-RF
                break;
            end
        end
        paris(num+1,1)=dis_1;
        paris(num+1,2)=dis_2;
        paris(num+1,3)=0;
        num=num+1;
    end
    %% for skilled forgeries
    for j = 1 : rat_sf
        % dis_1=unidrnd(gallery_num)+gallery_num; % from skilled forgeries
        dis_1=unidrnd(length(label)-Gstop_idx)+Gstop_idx; % accessing skilled forgeries
        while(1)
            dis_2=unidrnd(Gstop_idx); % accessing genuines
            if label(dis_1)==label(dis_2) % G-SF
                break;
            end
        end
        paris(num+1,1)=dis_1;
        paris(num+1,2)=dis_2;
        paris(num+1,3)=0;
        num=num+1;
    end
end
end