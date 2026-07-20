% A 10by10 SPD matrix has a total of 2^(10)-(10+1)=1013 principal submatrices of sizes m=2,...,10. 
% In particular there are |CKm|=combinations principal submatrices of size mbym where
% CKm is the matrix whose rows are the indices that give the mbym principal submatrices. 
function PSidx = principal_submatrices(dataG, sizes)

    PSidx = cell(1, length(sizes));
    n=10;
    sizevec=1:n;

    [writers_no, sig_no] = size(dataG);

    % CK# stands for all possible combinations of [1:n] over # 
    CK2=nchoosek(sizevec,2); %This matrix has 45 rows since "combinations 10 choose 2" =45
    CK3=nchoosek(sizevec,3); %This matrix has 180 rows since "combinations 10 choose 3" =120
    CK4=nchoosek(sizevec,4); %This matrix has 210 rows since "combinations 10 choose 4" =210
    CK5=nchoosek(sizevec,5); %This matrix has 504 rows since "combinations 10 choose 2" =252
    CK6=nchoosek(sizevec,6); %This matrix has 210 rows since "combinations 10 choose 6" =210
    CK7=nchoosek(sizevec,7); %This matrix has 180 rows since "combinations 10 choose 7" =120
    CK8=nchoosek(sizevec,8); %This matrix has 45 rows since "combinations 10 choose 8" =45
    CK9=nchoosek(sizevec,9); %This matrix has 10 rows since "combinations 10 choose 9" =10
    %These are a total of 1012 principal submatrices (+1 10by10 is 1013).
    
    %Next we create arrays to store mbym determinants.
    %Initialize array of determinants (We will pick max value) 
    det2=zeros(1,max(size(CK2)));
    det3=zeros(1,max(size(CK3)));
    det4=zeros(1,max(size(CK4)));
    det5=zeros(1,max(size(CK5)));
    det6=zeros(1,max(size(CK6)));
    det7=zeros(1,max(size(CK7)));
    det8=zeros(1,max(size(CK8)));
    det9=zeros(1,max(size(CK9)));
        
    %Initialize matrix of indices. INDEX_m(i,j) is a matrix containing the indices
    %that maximize the determinant of mbym principal submatrix for the i-writer.
    INDEX_2=zeros(sig_no,2);
    INDEX_3=zeros(sig_no,3);
    INDEX_4=zeros(sig_no,4);
    INDEX_5=zeros(sig_no,5);
    INDEX_6=zeros(sig_no,6);
    INDEX_7=zeros(sig_no,7);
    INDEX_8=zeros(sig_no,8);
    INDEX_9=zeros(sig_no,9);
    
    for writer_no=1:writers_no
        for sig=1:sig_no
            A=dataG(writer_no,sig).cov(1).cov;
            
            for k=1:max(size(CK2))
                det2(k)=det(A(CK2(k,:),CK2(k,:)));
            end
            [max_det2,j2]=max(det2);
            INDEX_2(sig,:)=CK2(j2,:);
            results.MODE_2(writer_no,:)=mode(INDEX_2);
            
            
            for k=1:max(size(CK3))
                det3(k)=det(A(CK3(k,:),CK3(k,:)));
            end
            [max_det3,j3]=max(det3);
            INDEX_3(sig,:)=CK3(j3,:);
            MODE_3=mode(INDEX_3);
            results.MODE_3(writer_no,:)=mode(INDEX_3);
            
            
            for k=1:max(size(CK4))
                det4(k)=det(A(CK4(k,:),CK4(k,:)));
            end
            [max_det4,j4]=max(det4);
            INDEX_4(sig,:)=CK4(j4,:);
            MODE_4=mode(INDEX_4);
            results.MODE_4(writer_no,:)=mode(INDEX_4);
            
            
            for k=1:max(size(CK5))
                det5(k)=det(A(CK5(k,:),CK5(k,:)));
            end
            [max_det5,j5]=max(det5);
            INDEX_5(sig,:)=CK5(j5,:);
            MODE_5=mode(INDEX_5);
            results.MODE_5(writer_no,:)=mode(INDEX_5);
            
            
            for k=1:max(size(CK6))
                det6(k)=det(A(CK6(k,:),CK6(k,:)));
            end
            [max_det6,j6]=max(det6);
            INDEX_6(sig,:)=CK6(j6,:);
            MODE_6=mode(INDEX_6);
            results.MODE_6(writer_no,:)=mode(INDEX_6);
            
    
            for k=1:max(size(CK7))
                det7(k)=det(A(CK7(k,:),CK7(k,:)));
            end
            [max_det7,j7]=max(det7);
            INDEX_7(sig,:)=CK7(j7,:);
            MODE_7=mode(INDEX_7);
            results.MODE_7(writer_no,:)=mode(INDEX_7);
            
    
            for k=1:max(size(CK8))
                det8(k)=det(A(CK8(k,:),CK8(k,:)));
            end
            [max_det8,j8]=max(det8);
            INDEX_8(sig,:)=CK8(j8,:);
            results.MODE_8(writer_no,:)=mode(INDEX_8);
            
            
            for k=1:max(size(CK9))
                det9(k)=det(A(CK9(k,:),CK9(k,:)));
            end
            [max_det9,j9]=max(det9);
            INDEX_9(sig,:)=CK9(j9,:);
            results.MODE_9(writer_no,:)=mode(INDEX_9);
        end
    
    end

    for i=1:length(sizes)
        PSidx{i} = mode(results.(sprintf("MODE_%d",sizes(i))));
    end
end
