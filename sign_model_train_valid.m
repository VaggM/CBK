function learned_model = sign_model_train_valid( ...
    traindbmat,tr_val_param, rat_gen,rat_rf,rat_sf,batch_mult, ...
    DEVW,learn_nosigs,learn_nosigsf, ...
    segments_index_train,model,metric,output_path)

% Development - Learning = {Training & Validate} with the 1x1 window - cov
% traindbmat: the struct which contains the dataset.
% tr_val_param: train split value.
% rat_gen: percentage of genuines on mini-batches.
% rat_rf: percentage of random forgeries on mini-batches.
% rat_sf: percentage of skilled forgeries on mini-batches.
% batch_mult: batch size multiplier.
% DEVW: Development Writers set for the learning stage.
% learn_nosigs: Number of forgery signatures per writer.
% segments_index_train: Slice indices used during learning.
% model: Model structure that saves metadata and results.
% metric: Metric used to compare embedding vectors.
% output_path: Path to save output files.

%% Setup the Positive and Negative Classes

indx_tr_samples = sort(randsample(learn_nosigs,round(tr_val_param*learn_nosigs)));
indx_tr_samplesF = sort(randsample(learn_nosigsf,round(tr_val_param*learn_nosigsf)));

[CDevGT, CDevFT, ...
    labelsG, labelsF, ...
    CDevGV, CDevFV, ...
    labelsGV, labelsFV] = create_omega_sets( ...
    traindbmat, ...
    DEVW, learn_nosigs, learn_nosigsf, ...
    segments_index_train, ...
    indx_tr_samples, indx_tr_samplesF ...
    );

% Setting up the Learning set
CdevG_F_T=cat(3,CDevGT,CDevFT);
labelsG_F_T=[labelsG;labelsF];
stop_idx_TG=size(CDevGT,3);

% Setting up the Validation set
CdevG_F_V=cat(3,CDevGV,CDevFV);
labelsG_F_V=[labelsGV;labelsFV];
stop_idx_VG=size(CDevGV,3);

%  Setting up the batch size
% % final size of mini-batch
batchsize=(rat_gen+rat_rf+rat_sf)*batch_mult;

% Run the learning algorithm

% Just a change in the name of variables..
C_Train=CdevG_F_T;
C_Val=CdevG_F_V;
inputspd_size=size(C_Train,1);

% Init model state

L = length(model.L);
Lindex = model.L;
K = model.K;
batches = model.batches;

% Run atoms cases
for atom_case=1:length(model.k)

    Amax=-inf;

    %% CBK Model Initialization
    k = model.k(atom_case);

    fprintf('Learning k=%d, K=%d\n', k, K)
    
    fprintf('Initializing model variables\n')
    BB = kmeans_KarcherMean(CDevGT, K);
    B = cell(size(BB,3),1);
    for z = 1:size(BB,3)
        B{z} = squeeze(BB(:,:,z));
    end
    B = B';
    alpha = ones(K, 1);
    beta = ones(K, 1);
    M = eye(k*L);
    
    params.lam = 0.5;
    params.zs = 2;
    params.zd = 20;
    params.iter = 2;
    params.internal_iter = 20;
    params.m0 = eye(k*L);
    
    params.K = K;
    params.k = k;
    params.L = L;
    params.Lindex = Lindex;
    
    %% Batch Training
    
    fprintf('Ready to train\n')

    % train stats
    learnLoss = zeros(2*batches,1);
    sim_d = zeros(batches,1);
    dis_d = zeros(batches,1);
    atomIndicator = zeros(K,batches);

    fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    
    for i = 1:batches

        figure_name = sprintf("CBK Batch[%d] L=%d, k=%d, K=%d", i, L, k, K);

        % File my_generatepairs creates pairs of covariances for the
        %  mini batch according to the batchsize,rat_gen,rat_rf,rat_sf
        %  parameters
        fprintf('Training batch[%d]\n', i)

        pairs=gen_pairs(labelsG_F_T, batchsize, rat_gen, rat_rf, rat_sf, stop_idx_TG);
    
        % initializing the mini-batches
        batch_x1=zeros(size(C_Train,1),size(C_Train,2),size(pairs,1));
        batch_x2=zeros(size(C_Train,1),size(C_Train,2),size(pairs,1));
    
        % creating the mini-batches.
        % j-index is the index of the j-th pair of the mini-batch
        % batch_x1 is the first part of the j-th pair
        % batch_x2 is the second part of the j-th pair
        for j=1:size(pairs,1)
            batch_x1(:,:,j)=C_Train(:,:,pairs(j,1));
            batch_x2(:,:,j)=C_Train(:,:,pairs(j,2));
        end
    
        % Train the model with: the covariance pairs batch_x1 batch_x2 & the labels of the mini-batch indexed by pairs(:,3)

        try

            [B, alpha, beta, M, batchloss, sim_d(i), dis_d(i), atomIndicator(:,i)] = CBK(batch_x1, batch_x2, pairs(:,3), B, alpha, beta, M, params);

        catch
            fprintf("\tError occured during training\n")
            sim_d(i) = sim_d(i-1);
            dis_d(i) = dis_d(i-1);
            atomIndicator(:,i) = zeros(K,1);
        end

        learnLoss(i*2-1:i*2) = batchloss;

        %% plot some graphs for the training stage
        figure(1);
        sgtitle(figure_name)
        subplot(221), plot(learnLoss(1:2*i)), hold on, plot(2:2:2*i, learnLoss(2:2:2*i),'*r'), hold off, title('Training Loss');
        subplot(222), bar(atomIndicator(:,1:i),'stacked'), title('Atom Usage');
        subplot(223), plot(sim_d(1:i),'b'), hold on, plot(dis_d(1:i),'r'), hold off, title('Dis-Similar scores');
        drawnow();

        %% Validation procedure, every 20 iterations.
        %  Calculate similarity on the validation SET labelsG_F_V & C_t=CdevG_F_V;
        
        pidx=0;
        didx=0;
        
        % ------------ Place here the start of the validation code ---------------
        Gval_length_per_writer=learn_nosigs-length(indx_tr_samples);
        GGpairs=nchoosek(1:Gval_length_per_writer,2);

        SFval_length_per_writer=learn_nosigsf-length(indx_tr_samplesF);

        CombArray=table2array(combinations(1:Gval_length_per_writer,1:SFval_length_per_writer));
        SFpairs=CombArray(randsample(length(CombArray),length(GGpairs)),:);
     
        % Positive GV-GV pairs - omega+
        for idxwr=1:length(DEVW)
            for kk=1:length(GGpairs)
                % GG pairs
                T1G=CDevGV(:,:,(idxwr-1)*Gval_length_per_writer+GGpairs(kk,1));
                T2G=CDevGV(:,:,(idxwr-1)*Gval_length_per_writer+GGpairs(kk,2));

                % USE THE MODEL with T1G, T2G to evaluate their GG
                % distance (dm). 

                pidx=pidx+1;

                val_simdis(pidx)= CBK_dist_split(T1G, T2G, B, alpha, beta, M, k, K, Lindex);

            end
        end

        % negative GV-SFV pairs, omega-
        for idxwr=1:length(DEVW)
            for kk=1:length(SFpairs)

                % GSF pairs
                T1G=CDevGV(:,:,(idxwr-1)*Gval_length_per_writer+SFpairs(kk,1));
                T2SF=CDevFV(:,:,(idxwr-1)*SFval_length_per_writer+SFpairs(kk,2));


                % USE THE MODEL with T1G, T2SF to evaluate their G-SF
                % distance (dm). 

                didx=didx+1;

                val_difdis(didx) = CBK_dist_split(T1G, T2SF, B, alpha, beta, M, k, K, Lindex);

            end
        end

        % ------------ End of validation code ---------------
        % ROC analysis
        [far,pd,~,auc] = perfcurve([ones(length(val_simdis),1); -ones(length(val_difdis),1)]',[val_simdis'; val_difdis']',-1);

        if auc > Amax

            Amax = auc;

            % Best model
            learned_model.B{atom_case} = B;

            if metric=="ABLD"
                learned_model.alpha{atom_case} = alpha;
                learned_model.beta{atom_case} = beta;
            end

            learned_model.M{atom_case} = M;

            imax=i;

        end
        
        % PLOT ROC validation set 
        subplot(224), plot(far,pd); grid on; grid('minor'),
        title(['ROC - AUC=', num2str(auc) ', max=', num2str(Amax) ])
        drawnow();
        disp(['curAUC[', num2str(i), ']=', num2str(auc), ])
        disp(['maxAUC[', num2str(imax), ']=', num2str(Amax)])

        clear val_simdis val_difdis

        fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
        
    end
    
    %% Learning stats
    learned_model.stats_train{atom_case}.loss = learnLoss;
    learned_model.stats_train{atom_case}.sim_d = sim_d;
    learned_model.stats_train{atom_case}.dis_d = dis_d;
    learned_model.stats_train{atom_case}.atom_idx = atomIndicator;

    learned_model.stats_valid{atom_case}.far = far;
    learned_model.stats_valid{atom_case}.pd = pd;
    learned_model.stats_valid{atom_case}.auc = auc;
    
    figure_file = sprintf("./%s/%s.png", output_path, figure_name);
    saveas(gcf, figure_file);

end

