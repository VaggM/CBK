%% =========================================================================
%  Copyright (c) 2026, E. Mitikas, C. Chorianopoulos, & E. Zois.
%  All rights reserved.
%
%  This software provides demonstration and reproducibility for the paper:
%  Mitikas, E., Chorianopoulos, C., & Zois, E. (2026). Embedding Riemannian 
%  Collective Background Knowledge for Offline Signature Verification. 
%  Machine Learning and Knowledge Extraction, 8(7), 208.
%  https://doi.org/10.3390/make8070208
%
%  EXPERIMENTAL PROTOCOL CITATIONS:
%  If you make use of this experimental protocol, please refer to:
%  1. Mitikas et al., MAKE 2026, DOI: 10.3390/make8070208
%  2. Vasilakis et al., Appl. Sci. 2025, DOI: 10.3390/app15137015
%  3. Zois et al., IEEE TIFS 2024, DOI: 10.1109/TIFS.2023.3333681
%  4. Giazitzis & Zois, ESWA 2025, DOI: 10.1016/j.eswa.2024.125470
%  5. Giazitzis & Zois, IEEE/CVF WACV 2024, DOI: 10.1109/WACV57701.2024.00619
%  6. Giazitzis et al., ICDAR 2024, DOI: 10.1007/978-3-031-70536-6_13
% =========================================================================
%  EXTERNAL DEPENDENCIES & ACKNOWLEDGMENTS:
%
%  1. Manopt v8.0 (Released July 5, 2024)
%     Citation: Boumal, N., Mishra, B., Absil, P.-A., & Sepulchre, R. (2014).
%     Manopt, a Matlab Toolbox for Optimization on Manifolds. JMLR, 15, 1455-1459.
%     Info: https://www.manopt.org
%
%  2. Karcher Mean algorithms
%     Citations: 
%     - Cheng et al., ECCV 2012 (Mean Diffusion Tensor & DTI Segmentation)
%     - Ho et al., AISTATS 2013 (Recursive Karcher Expectation Estimators)
%     - Salehian et al., ICCV 2013 (Recursive Estimation of Stein Center)
%
%  3. Discriminative Alpha-Beta Divergence Framework
%     Copyright (c) 2017, Panagiotis Stanitsas and Anoop Cherian.
%     Citation: Cherian et al., Learning Discriminative Alpha-Beta Divergence 
%     for Positive Definite Matrices, ICCV, 2017.
%     Note: Distributed under the BSD 3-Clause License. See root LICENSE file.
%
%  4. Signature Verification Evaluation Tools
%     Citations:
%     - Hafemann et al., Pattern Recognition 2017 (Feature Learning via DCNNs)
%     - Hafemann et al., IJDAR 2018 (Fixed-sized representation learning)
%     Repository: https://github.com/luizgh/sigver_wiwd
%     Note: Distributed under the BSD 2-Clause License. See root LICENSE file.
% =========================================================================

clear, clc

% This software provides demonstration and reproducibility
% for the L5 and L5G scenarios using CEDAR in an F_{IL} setting

CEDAR_filename = './datasets/CEDAR.mat';

learndb='CEDAR';

learn_mat = load(CEDAR_filename);
learn_nowriters=55;
learn_nosigs=24; 
learn_nosigsf=24;

testdb='CEDAR';

test_mat = load(CEDAR_filename);
test_nowriters=55;
test_nosigs=24; 
test_nosigsf=24;

% Parameters for mini-batch formation
rat_rf=0;
rat_gen=10;
rat_sf=rat_gen-rat_rf;

% Parameters
repetitions = 5;
batch_mult=50;

test_type = "L5"; % L5 or L5G
metric = "ABLD";

% Models
K40.K = 40;
K40.k = [40];

models = [K40];

% Output path
KK = zeros(1, length(models));
for i=1:length(models)
    KK(i) = models(i).K;
end
output_path = sprintf("runtime_results_intra_%s_%s_%s_ratrf_%d_K_%s", metric, test_type, learndb, rat_rf, num2str(KK));

%% Start logging
if ~isfolder(output_path)
    mkdir(output_path);
end

diary(sprintf('./%s/logs.txt', output_path));

%% Set Test Type
new_models = [];
for i=1:length(models)

    model = models(i);

    model.batches = 10;

    if strcmp(test_type,"L5")
        model.L_init = {1:10, 1:7, 2:8, 3:9, 4:10};
        model.PS = [];

    elseif strcmp(test_type,"L5G")
        model.L_init = {1:10};
        model.PS = 5:8;

    end

    new_models = [new_models, model];
end
models = new_models;

%% Load functions
addpath(genpath(pwd()));

% Size of initial SPD manifold. 10x10
szcov=size(learn_mat.(learndb).G(1,1).cov(1).cov,1);

% 10x10 covariance
idx=1:szcov;

% Internal development writers as a percentage of the entire population of
% writers. i.e: 50%
percofDEVW=0.5;
% Training with 70% of samples of each DEVW and validating with 30%
tr_val_param=0.7;
% number of development writers
NODEVW=round(learn_nowriters*percofDEVW);


disp('CBK Model Learning & Evalutation ')
fprintf('Train: %s, Test: %s, rat_rf: %d\n', learndb, testdb, rat_rf)
fprintf('Metric: %s, Mode: %s\n', metric, test_type)

%%-----BEGIN OF TRAINING ----------
% Development writers
for rep=1:repetitions

    DEVW = sort(randsample(learn_nowriters,NODEVW))';
    TESTW = 1:learn_nowriters;
    TESTW(DEVW) = [];

    % fold rep A

    disp(['Learning repetition No.', num2str(rep), 'A'])
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    
    %%---- Evaluate training parameters W, M, M0, A
    for i=1:length(models)

        % Check if General Principal Submatrices are used
        if ~isempty(models(i).PS)
            L_SF = principal_submatrices(learn_mat.(learndb).G(DEVW,:), models(i).PS);
            models(i).L = [models(i).L_init, L_SF];
        else
            models(i).L = models(i).L_init;
        end

        % Use Algorithm 2: Development of the Model \Theta
        models(i).(sprintf("rs%d",rep)) = sign_model_train_valid( ...
            learn_mat.(learndb), tr_val_param, rat_gen, rat_rf, rat_sf, batch_mult, ...
            DEVW, learn_nosigs, learn_nosigsf, ...
            1, models(i), metric, output_path);

        % Save used DEVW and L for preview later
        models(i).(sprintf("rs%d",rep)).DEVW = DEVW;
        models(i).(sprintf("rs%d",rep)).L = models(i).L;
    end
    
    save(sprintf('./%s/models.mat', output_path), "models")
    
    %%----- TESTING ----------
    for i=1:length(models)
        for j=1:length(models(i).k)

            learned_model.B = models(i).(sprintf("rs%d",rep)).B{j};
            learned_model.alpha = models(i).(sprintf("rs%d",rep)).alpha{j};
            learned_model.beta = models(i).(sprintf("rs%d",rep)).beta{j};
            learned_model.M = models(i).(sprintf("rs%d",rep)).M{j};

            learned_model.k = models(i).k(j);
            learned_model.K = models(i).K;
            learned_model.L = models(i).L;

            fprintf("Testing rs k=%d K=%d\n", models(i).k(j), models(i).K)

            models(i).(sprintf("rs%d",rep)).stats_test{j} = sign_model_testing( ...
                test_mat.(testdb), 10, 1:14, TESTW, test_nosigs, test_nosigsf, ...
                learned_model, metric);
            
        end
    end

    save(sprintf('./%s/models.mat', output_path), "models")

    % fold rep B

    disp(['Learning repetition No.', num2str(rep), 'B'])
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    for i=1:length(models)

        if ~isempty(models(i).PS)
            L_SF = principal_submatrices(test_mat.(testdb).G(TESTW,:), models(i).PS);
            models(i).L = [models(i).L_init, L_SF];
        else
            models(i).L = models(i).L_init;
        end

        % Use Algorithm 2: Development of the Model \Theta
        models(i).(sprintf("irs%d",rep)) = sign_model_train_valid( ...
            test_mat.(testdb), tr_val_param, rat_gen, rat_rf, rat_sf, batch_mult, ...
            DEVW, test_nosigs, test_nosigsf, ...
            1, models(i), metric, output_path);

        % Save used L for preview later
        models(i).(sprintf("irs%d",rep)).DEVW = TESTW;
        models(i).(sprintf("irs%d",rep)).L = models(i).L;
    end

    save(sprintf('./%s/models.mat', output_path), "models")

    %%----- TESTING ----------
    for i=1:length(models)
        for j=1:length(models(i).k)

            learned_model.B = models(i).(sprintf("irs%d",rep)).B{j};
            learned_model.alpha = models(i).(sprintf("irs%d",rep)).alpha{j};
            learned_model.beta = models(i).(sprintf("irs%d",rep)).beta{j};            
            learned_model.M = models(i).(sprintf("irs%d",rep)).M{j};

            learned_model.k = models(i).k(j);
            learned_model.K = models(i).K;
            learned_model.L = models(i).L;

            fprintf("Testing irs k=%d K=%d\n", models(i).k(j), models(i).K)

            models(i).(sprintf("irs%d",rep)).stats_test{j} = sign_model_testing( ...
                learn_mat.(learndb),10,1:14,DEVW,learn_nosigs,learn_nosigsf, ...
                learned_model, metric);
        end
    end

    save(sprintf('./%s/models.mat', output_path), "models")

end

diary off;
