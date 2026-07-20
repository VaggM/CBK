function [CDevGT, CDevFT, ...
    labelsG, labelsF, ...
    CDevGV, CDevFV, ...
    labelsGV, labelsFV] = create_omega_sets( ...
    traindbmat, ...
    DEVW, learn_nosigs, learn_nosigsf, ...
    segments_index_train, ...
    indx_tr_samples, indx_tr_samplesF ...
    )

    %% Build training set: Positive Class (omega+)
    kk=0;
    kkx=0;
    labelsG=[];
    labelsGV=[];
    for SignerIDidx=1:length(DEVW)
        for j=1:learn_nosigs
            % The j-th sample could be either for training of validation
            if sum(j==indx_tr_samples)==1 % training
                labelsG=[labelsG;SignerIDidx];
                % segments (well now we use only one segment - 1x1)
                tmp=[];
                for p=1:length(segments_index_train)
                    segm=traindbmat.G(DEVW(SignerIDidx),j).cov(segments_index_train(p)).cov;
                    segm=segm+diag(ones(1,size(segm,1))*0.001*trace(segm));
                    tmp=blkdiag(tmp,segm);
                end
                kk=kk+1;
                CDevGT(:,:,kk)=tmp;
                % small edit, just in case that the covariance is semi-positive definite
                Ixx=diag(ones(1,size(CDevGT(:,:,kk),1))*0.001*trace(CDevGT(:,:,kk)));
                % now is positive definite
                CDevGT(:,:,kk)=CDevGT(:,:,kk)+Ixx;
            else % validation
                tmp=[];
                for p=1:length(segments_index_train)
                    segm=traindbmat.G(DEVW(SignerIDidx),j).cov(segments_index_train(p)).cov;
                    segm=segm+diag(ones(1,size(segm,1))*0.001*trace(segm));
                    tmp=blkdiag(tmp, segm);
                end
                labelsGV=[labelsGV;SignerIDidx];
                kkx=kkx+1;
                CDevGV(:,:,kkx)=tmp;
                Ixx=diag(ones(1,size(CDevGV(:,:,kkx),1))*0.001*trace(CDevGV(:,:,kkx)));
                CDevGV(:,:,kkx)=CDevGV(:,:,kkx)+Ixx;
            end
        end
    end
    
    %% Build training set: Negative Class (omega-) 
    kk1=0;
    kk2=0;
    labelsF=[];
    labelsFV=[];
    for SignerIDidx=1:length(DEVW)
        for j=1:learn_nosigsf % 1:learn_nosigsf
            if sum(j==indx_tr_samplesF)==1
                labelsF=[labelsF;SignerIDidx];
                tmpF=[];
                for p=1:length(segments_index_train)
                    segmF=traindbmat.F(DEVW(SignerIDidx),j).cov(segments_index_train(p)).cov;
                    segmF=segmF+diag(ones(1,size(segmF,1))*0.001*trace(segmF));
                    tmpF=blkdiag(tmpF,segmF);
                end
                kk1=kk1+1;
                CDevFT(:,:,kk1)=tmpF;
                Ixx=diag(ones(1,size(CDevFT(:,:,kk1),1))*0.001*trace(CDevFT(:,:,kk1)));
                CDevFT(:,:,kk1)=CDevFT(:,:,kk1)+Ixx;
            else
                tmpF=[];
                for p=1:length(segments_index_train)
                    segmF=traindbmat.F(DEVW(SignerIDidx),j).cov(segments_index_train(p)).cov;
                    segmF=segmF+diag(ones(1,size(segmF,1))*0.001*trace(segmF));
                    tmpF=blkdiag(tmpF,segmF);
                end
                labelsFV=[labelsFV;SignerIDidx];
                kk2=kk2+1;
                CDevFV(:,:,kk2)=tmpF;
                Ixx=diag(ones(1,size(CDevFV(:,:,kk2),1))*0.001*trace(CDevFV(:,:,kk2)));
                CDevFV(:,:,kk2)=CDevFV(:,:,kk2)+Ixx;
            end
        end
    end

end