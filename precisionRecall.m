function [ optPrec, optRec, opt_Lapl] = precisionRecall_gmm(true_Laplacian, learned_Laplacian, varargin)
%returns the precision and recall of edges recovery, and the thresholded 
%Laplacian when the number of learned edges is around the number of 
%original edges
optPrec=0;
optRec=0;
opt_Lapl = zeros(size(true_Laplacian));
verbose = 0;
if(length(varargin)>0)
    verbose = varargin{1};
end

first_crossing=1;
for i = 1 : 300
    Laplacian = learned_Laplacian;
    mytol = i/1000;
    Laplacian(Laplacian>-mytol & Laplacian<0.5) = 0;

    realLowTri = logical(tril(true_Laplacian,-1)~=0);
    estimLowTri = logical(tril(Laplacian,-1)~=0);
    
    % Total number of estimated edges
    num_of_edges = sum(sum(estimLowTri));
    real_edges = sum(sum(realLowTri));
    
    if(num_of_edges > 0)
    
        % Precision - fraction of estimated edges that are correct
        precision = sum(sum(realLowTri.*estimLowTri)) / num_of_edges;

        % Recall - fraction of correct edges that are estimated
        recall = sum(sum(realLowTri.*estimLowTri)) / real_edges;
    else
        precision = 0;
        recall = 0;
    end
    
    myprec(i) = precision;
    myrec(i) = recall;
    myedges(i) = num_of_edges;
    
    if (num_of_edges<=real_edges && first_crossing)
        %disp(i);
        optPrec=precision;
        optRec=recall;
        first_crossing = 0;
        opt_Lapl = Laplacian;
    end
end

x = find(myedges==0);
if(x)
    maxInd = x(1);
    myedges = myedges(1:maxInd);
    myprec = myprec(1:maxInd);
    myrec = myrec(1:maxInd);
else
    maxInd = length(myedges);
end

%clf;
if (verbose)
    hold on;
    plot(myprec);
    plot(myrec);
    plot(myedges/real_edges);
    xlim([1 maxInd]);
    ylim([0 1.1]);
    %legend('Precision','Recall','Edges');
end

end

