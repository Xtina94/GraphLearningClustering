%%
clear all
close all
addpath(genpath(pwd))

load DataSetDorina.mat
load ComparisonDorina.mat
ds_name = 'Dorina';
path = 'C:\Users\cryga\Documents\GitHub\GraphLearningClustering\Results';
X = TrainSignal;

%load data in variable X here (it should be a matrix #nodes x #signals)
param.N = size(X,1); % number of nodes in the graph
param.S = 4;  % number of subdictionaries 
param.J = param.N * param.S; % total number of atoms 
param.K = [20 20 20 20]; % polynomial degree of each subdictionary
param.c = 1; % spectral control parameters
param.epsilon = 0.05;%0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1;%1e-2; % polynomial regularizer paremeter
param.y = X; %signals
param.y_size = size(param.y,2);

%% generate dictionary polynomial coefficients from heat kernel
% % % param.t(1) = 2; %heat kernel coefficients
% % % param.t(2) = 1; %this heat kernel will be inverted to cover high frequency components
% % % param.alpha = generate_coefficients(param);
% % % disp(param.alpha);

for i = 1:param.S
    param.alpha{i} = comp_alpha((i-1)*(max(param.K)+1) + 1:(max(param.K)+1)*i);
end

%% initialise learned data
param.T0 = 6; %sparsity level (# of atoms in each signals representation)
[param.Laplacian, learned_W] = init_by_weight(param.N);
[learned_dictionary, param] = construct_dict(param);
alpha = 2; %gradient descent parameter, it decreases with epochs
elapsed_time = zeros(20,1);
for big_epoch = 1:20 
    tic
    if big_epoch == 1
        %% optimise with respect to x
        disp(['Epoch... ',num2str(big_epoch)]);
        x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
        
        %% optimise with respect to W
        maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
        beta = 10^(-2); %graph sparsity penalty
        old_L = param.Laplacian;
        [param.Laplacian, learned_W] = update_graph(x, alpha, beta, maxEpoch, param, learned_W);
        [learned_dictionary, param] = construct_dict(param);
        alpha = alpha*0.985; %gradient descent decreasing
        
        %% %%%%%%% Find the two communities separation through fiedler's eigenvector %%%%%%%
        %optimise with regard to x
        disp(['Epoch... ',num2str(big_epoch)]);
        x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
        
        %optimise with regard to W
        maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
        beta = 10^(-2); %graph sparsity penalty
        old_L = param.Laplacian;
        [param.Laplacian, learned_W] = update_graph(x, alpha, beta, maxEpoch, param, learned_W);
        [learned_dictionary, param] = construct_dict(param);
        alpha = alpha*0.985; %gradient descent decreasing
        
        % extract eigenvectors
        N = param.N;
        [V,DD] = eigs(param.Laplacian,3,'SA'); % find the 3 smallest eigenvalues and corresponding eigenvectors
        v1 = V(:,2)/norm(V(:,2)); % Fiedler's vector
        %         v2 = V(:,3)/norm(V(:,3));
        
        % Separate into two communities
        % sweep wrt the ordering identified by v1
        % reorder the adjacency matrix
        [v1s,pos] = sort(v1);
        learned_W1 = learned_W(pos,pos);
        
        % evaluate the conductance measure
        a = sum(triu(learned_W1));
        b = sum(tril(learned_W1));
        d = a+b;
        D = sum(d);
        assoc = cumsum(d);
        assoc = min(assoc,D-assoc);
        cut = cumsum(b-a);
        conduct = cut./assoc;
        conduct = conduct(1:end-1);
        % show the conductance measure
        figure('Name','Conductance')
        plot(conduct,'x-')
        grid
        title('conductance')
        
        % identify the minimum -> threshold
        [~,mpos] = min(conduct);
        threshold = mean(v1s(mpos:mpos+1));
        disp(['Minimum conductance: ' num2str(conduct(mpos))]);
        disp(['   Cheeger''s upper bound: ' num2str(sqrt(2*DD(2,2)))]);
        disp(['   # of links: ' num2str(D/2)]);
        disp(['   Cut value: ' num2str(cut(mpos))]);
        disp(['   Assoc value: ' num2str(assoc(mpos))]);
        disp(['   Community size #1: ' num2str(mpos)]);
        disp(['   Community size #2: ' num2str(N-mpos)]);
        learned_W1 = learned_W(pos(1:mpos),pos(1:mpos));
        learned_W2 = learned_W(pos(mpos+1:N),pos(mpos+1:N));
        param.pos = pos;
        param.mpos = mpos;
        param.pos1 = param.pos(1:param.mpos);
        param.pos2 = param.pos(param.mpos+1:end);
        % Construct the two subLaplacians
        param.L1 = diag(sum(learned_W1,2)) - learned_W1; % combinatorial Laplacian
        param.L2 = diag(sum(learned_W2,2)) - learned_W2; % combinatorial Laplacian
        param.Laplacian1 = (diag(sum(learned_W1,2)))^(-1/2)*param.L1*(diag(sum(learned_W1,2)))^(-1/2); % normalized Laplacian
        param.Laplacian2 = (diag(sum(learned_W2,2)))^(-1/2)*param.L2*(diag(sum(learned_W2,2)))^(-1/2); % normalized Laplacian        
        % Separate the two dictionaries
        [learned_dictionary1, param ] = construct_dict_comm(param,mpos,1);
        [learned_dictionary2, param ] = construct_dict_comm(param,N-mpos,2);
        
    else
        %% optimise with respect to x
        disp(['Epoch... ',num2str(big_epoch)]);
        x1 = OMP_non_normalized_atoms(learned_dictionary1,param.y(param.pos(1:param.mpos),:), param.T0);
        x2 = OMP_non_normalized_atoms(learned_dictionary2,param.y(param.pos(param.mpos+1:N),:), param.T0);
        
        %% optimise with respect to W
        maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
        beta = 10^(-2); %graph sparsity penalty
        old_L1 = param.Laplacian1;
        old_L2 = param.Laplacian2;
        [param.Laplacian1, learned_W1] = update_graph_comm(x1, alpha, beta, maxEpoch, param, learned_W1, param.mpos, param.y(param.pos(1:param.mpos),:),1);
        [param.Laplacian2, learned_W2] = update_graph_comm(x2, alpha, beta, maxEpoch, param, learned_W2, param.N - param.mpos, param.y(param.pos(param.mpos+1:param.N),:),2);
        [learned_dictionary1, param ] = construct_dict_comm(param,param.mpos,1);
        [learned_dictionary2, param ] = construct_dict_comm(param,param.N-param.mpos,2);
        alpha = alpha*0.985; %gradient descent decreasing
    end
    toc
    elapsed_time(big_epoch) = toc;
end

avgTime = mean(elapsed_time);
disp(['The average elapsed time is: ',num2str(avgTime)]);

%%
%constructed graph needs to be tresholded, otherwise it's too dense
%fix the number of desired edges here at nedges

nedges = 2*29;
for i = 1:2
    eval(['final_Laplacian',num2str(i),' = treshold_by_edge_number(param.Laplacian',num2str(i),', nedges);']);
    eval(['final_W',num2str(i),' = learned_W',num2str(i),'.*(final_Laplacian',num2str(i),'~=0);']);
end

compl1 = zeros(length(param.pos1),length(param.pos2)); %zero padding to make the dimensions match
compl2 = zeros(length(param.pos2),length(param.pos1)); %zero padding to make the dimensions match

%% Reconstruct the dictionary
tmp = learned_dictionary1;
learned_dictionary1 = [];
for i = 1:param.S
    learned_dictionary1 = [learned_dictionary1 tmp(:,(i-1)*length(param.pos1)+1:i*length(param.pos1)) compl1];
end
tmp = learned_dictionary2;
learned_dictionary2 = [];
for i = 1:param.S
    learned_dictionary2 = [learned_dictionary2 compl2 tmp(:,(i-1)*length(param.pos2)+1:i*length(param.pos2))];
end
learned_dictionary = [learned_dictionary1; learned_dictionary2];

%% Reconstruct the adjacency matrix
tmp1 = [learned_W1 compl1];
tmp2 = [compl2 learned_W2];
learned_W = [tmp1; tmp2];

%% Estimate the final reproduction error
X = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*X,'fro')^2/size(TestSignal,2));
disp(['The total representation error of the testing signals is: ',num2str(errorTesting_Pol)]);

%% Save results
filename = [path,num2str(ds_name),'\Output_',num2str(ds_name),'.mat'];
learned_eigenVal = 0; %param.lambda_sym;
save(filename,'learned_dictionary','learned_W','X','learned_eigenVal');

%% Verify the results with the precision recall function
learned_L1 = diag(sum(learned_W1,2)) - learned_W1;
learned_Laplacian1 = (diag(sum(learned_W1,2)))^(-1/2)*learned_L1*(diag(sum(learned_W1,2)))^(-1/2);
learned_L2 = diag(sum(learned_W2,2)) - learned_W2;
learned_Laplacian2 = (diag(sum(learned_W2,2)))^(-1/2)*learned_L2*(diag(sum(learned_W2,2)))^(-1/2);

% Separate comp_W into the two subgraphs
comp_W1 = comp_W(param.pos(1:param.mpos),param.pos(1:param.mpos));
comp_W2 = comp_W(param.pos(param.mpos+1:end),param.pos(param.mpos+1:end));

comp_L1 = diag(sum(comp_W1,2)) - comp_W1;
comp_Laplacian1 = (diag(sum(comp_W1,2)))^(-1/2)*comp_L1*(diag(sum(comp_W1,2)))^(-1/2);
comp_L2 = diag(sum(comp_W2,2)) - comp_W2;
comp_Laplacian2 = (diag(sum(comp_W2,2)))^(-1/2)*comp_L2*(diag(sum(comp_W2,2)))^(-1/2);

[optPrec1, optRec1, opt_Lapl1] = precisionRecall(comp_Laplacian1, learned_Laplacian1);
[optPrec2, optRec2, opt_Lapl2] = precisionRecall(comp_Laplacian2, learned_Laplacian2);
filename = [path,num2str(ds_name),'\ouput_PrecisionRecall_',num2str(ds_name),'.mat'];
save(filename,'errorTesting_Pol','avgTime','opt_Lapl1','optPrec1','optRec1','opt_Lapl2','optPrec2','optRec2');
