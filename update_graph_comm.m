function [learned_Laplacian, learned_W] = update_graph_comm(x, alpha, beta, maxEpoch, param, learned_W, dimension, ref_y, p) 
%graph updating step by gradient descent

eye_N = eye(dimension);
for epoch =1:maxEpoch
     %compute signal estimation
    [learned_dictionary, param] = construct_dict_comm(param, dimension, p);
    estimated_y=learned_dictionary*x; 
    error(epoch) = sum(sum(abs(estimated_y-ref_y))) + beta*sum(sum(abs(learned_W)));
    %computing the gradient
    K=max(param.K);
    der_all_new = zeros(dimension, dimension);
    learned_D = diag(sum(learned_W,2));
    learned_D_powers{1} = learned_D^(-0.5);
    learned_D_powers{2} = learned_D^(-1);
    for s=1:param.S
        for k=0:K
            C=zeros(dimension,dimension);
            B=zeros(dimension,dimension);
            for r=0:k-1 
                eval(['A = learned_D_powers{1}*param.Laplacian_powers',num2str(p),'{k-r}*x((s-1)*dimension+1:s*dimension,:)*(estimated_y - ref_y)''*param.Laplacian_powers',num2str(p),'{r+1} * learned_D_powers{1};']);
                B=B+learned_D_powers{1}*learned_W*A*learned_D_powers{1};
                C=C-2*A';
                B=B+A*learned_W*learned_D_powers{2};
            end
            B = ones(size(B)) * (B .* eye_N);
            C = param.alpha{s}(k+1)*(C+B);
            der_all_new = der_all_new + C;
        end            
    end
    %adding the sparsity term gradient
    der_all_new = der_all_new +  beta*sign(learned_W); 
    %making derivative symmetric and removing the diag (that we don't want to change)
    der_sym = (der_all_new + der_all_new')/2 - diag(diag(der_all_new)); 
    
    %gradient descent, adjusting the weights with each step
    alpha = alpha * (0.1^(1/maxEpoch));
    %beta = beta * (10^(1/maxEpoch));
    learned_W = learned_W - alpha * der_sym;
    
    %producing a valid weight matrix
    learned_W(learned_W<0)=0;
    
    % combinatorial Laplacian
    learned_L = diag(sum(learned_W,2)) - learned_W;
    % normalized Laplacian
    eval(['param.Laplacian',num2str(p),' = (diag(sum(learned_W,2)))^(-1/2)*learned_L*(diag(sum(learned_W,2)))^(-1/2);']);

end
    eval(['learned_Laplacian = param.Laplacian',num2str(p),';']);
end

