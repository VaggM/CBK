% =========================================================================
%
%     The testing evaluation and EER calculation is based on the following work.
%
%     Signature Verification Evaluation Tools (`sigver_wiwd`)
%     Citations:
%     - Hafemann et al., Pattern Recognition 2017 (Feature Learning via DCNNs)
%     - Hafemann et al., IJDAR 2018 (Fixed-sized representation learning)
%     Repository: https://github.com/luizgh/sigver_wiwd
%     Note: Distributed under the BSD 2-Clause License. See root LICENSE file.
%
% =========================================================================

function RESNREF=sign_model_testing( ...
    testdbmat,NREFS,segments_index_test,TESTW,nosigs,nosigsf, ...
    model,metric)
% testdbmat: the struct which contains the dataset.
% NREFS: the number of reference samples of the testing stage. e.g.  NREFS=10.
% segments_index_test: the number of participating segments, e.g. segments_index_test=[1:14];
% TESTW: the testing subset.
% nosigs: the number of genuine signatures of a dataset. eg. 24
% nosigsf: the number of skilled forgery signatures of a dataset, e.g. 24.
% model: Model structure that saves metadata and results.
% metric: Metric used to compare embedding vectors.


B = model.B;
alpha = model.alpha;
beta = model.beta;
M = model.M;

k = model.k;
K = model.K;
Lindex = model.L;

%% TESTING WITH REFERENCES (NREF)
for SignerIDidx=1:length(TESTW)

    fprintf('\tTestWriter: %d/%d\n', SignerIDidx, length(TESTW))

    for urn=1:10 % 10 urns
        rng("default") % for demo purposes
        % the genuine reference subset.
        GREFS=sort(randsample(1:nosigs,NREFS));
        % REFERENCE SET X each writer of the TESTW subset.
        for iref=1:NREFS
            for kx=1:length(segments_index_test)
                segmREF=testdbmat.G(TESTW(SignerIDidx),GREFS(iref)).cov(segments_index_test(kx)).cov;
                segmREF=segmREF+diag(ones(1,size(segmREF,1))*0.001*trace(segmREF));
                X(:,:,iref,kx)=segmREF;
            end
        end
        % Genuine TEST Set (Questioned)
        GTEST=1:nosigs;GTEST(GREFS)=[];
        for igtest=1:length(GTEST)
            for kx=1:length(segments_index_test)
                segmQREF=testdbmat.G(TESTW(SignerIDidx),GTEST(igtest)).cov(segments_index_test(kx)).cov;
                segmQREF=segmQREF+diag(ones(1,size(segmQREF,1))*0.001*trace(segmQREF));
                QG(:,:,igtest,kx)=segmQREF;
            end
            for iref=1:NREFS
                for kx=1:length(segments_index_test)
                    T1G=X(:,:,iref,kx);     % 1st part of the pair. 
                    T2Q=QG(:,:,igtest,kx);  % 2nd part of the pair. 
                    
                    % Compute chosen metric
                    dm_GQG= CBK_dist_split(T1G, T2Q, B, alpha, beta, M, k, K, Lindex); 
                    
                    tmpsrepGGTESTlocREFS(igtest,iref,kx)=dm_GQG;
                end
                for stop_segm=1:length(segments_index_test)
                    if stop_segm==1
                        SSGGTESTlocREFS=squeeze(tmpsrepGGTESTlocREFS(igtest,iref,:));
                        srepGGTESTlocREFS(stop_segm).score(igtest,iref)=SSGGTESTlocREFS(stop_segm);
                    else
                        SSGGTESTlocREFS=sort(squeeze(tmpsrepGGTESTlocREFS(igtest,iref,:)));
                        srepGGTESTlocREFS(stop_segm).score(igtest,iref)=mean(SSGGTESTlocREFS(1:stop_segm));
                    end
                end
            end
        end

        for stop_segm=1:length(segments_index_test)
            srepGGTESTlocREFS(stop_segm).score=sort(srepGGTESTlocREFS(stop_segm).score,2);

            NREFscoreF1GG(stop_segm).NREFscoreF1GG=srepGGTESTlocREFS(stop_segm).score(:,1);
        end

        % Skilled forgery TEST test
        SFTEST=1:nosigsf;
        for isfest=1:length(SFTEST)
            for kx=1:length(segments_index_test)
                segmQREF=testdbmat.F(TESTW(SignerIDidx),isfest).cov(segments_index_test(kx)).cov;
                segmQREF=segmQREF+diag(ones(1,size(segmQREF,1))*0.001*trace(segmQREF));
                QF(:,:,isfest,kx)=segmQREF;
            end
            for iref=1:NREFS
                for kx=1:length(segments_index_test)
                    T1G=X(:,:,iref,kx);
                    T2QF=QF(:,:,isfest,kx);
                    
                    % Compute chosen metric

                    dm_GQF= CBK_dist_split(T1G, T2QF, B, alpha, beta, M, k, K, Lindex);

                    tmpsrepGSFTESTlocREFS(isfest,iref,kx)=dm_GQF;
                end
                for stop_segm=1:length(segments_index_test)
                    if stop_segm==1
                        SSGGTESTlocREFS=squeeze(tmpsrepGSFTESTlocREFS(isfest,iref,:));
                        srepGSFTESTlocREFS(stop_segm).score(isfest,iref)=SSGGTESTlocREFS(stop_segm);
                    else
                        SSGGTESTlocREFS=sort(squeeze(tmpsrepGSFTESTlocREFS(isfest,iref,:)));
                        srepGSFTESTlocREFS(stop_segm).score(isfest,iref)=mean(SSGGTESTlocREFS(1:stop_segm));
                    end
                end
            end
        end

        for stop_segm=1:length(segments_index_test)
            srepGSFTESTlocREFS(stop_segm).score=sort(srepGSFTESTlocREFS(stop_segm).score,2);

            NREFscoreF1GSF(stop_segm).NREFscoreF1GSF=srepGSFTESTlocREFS(stop_segm).score(:,1);
        end

        % ROC curves and associate error rates for NREF reference samples
        labels=[ones(1,length(NREFscoreF1GG(stop_segm).NREFscoreF1GG)) -ones(1,length(NREFscoreF1GSF(stop_segm).NREFscoreF1GSF))];
        for stop_segm=1:length(segments_index_test)
            [F1far,F1pd,~,~]=perfcurve(labels,[NREFscoreF1GG(stop_segm).NREFscoreF1GG ; NREFscoreF1GSF(stop_segm).NREFscoreF1GSF],-1);
            RESNREF(stop_segm).RESNREF(SignerIDidx,urn).SF1=min(mean([1-F1pd' ; F1far']));
        end
    end
end