%%
clear all
close all
addpath(genpath(pwd))

flag = 3;
switch flag
    case 4
        load DataSetDorina.mat
        load ComparisonDorina.mat
        X = TrainSignal;
        ds_name = 'Dorina';
        deg = 20;
        param.S = 4;  % number of subdictionaries 
    case 2
        load DataSetUber.mat
        load ComparisonUber.mat
        X = TrainSignal;
        ds_name = 'Uber';
        deg = 15;
        param.S = 2;  % number of subdictionaries 
    case 3
        load DataSetDoubleHeat.mat
        load ComparisonDoubleHeat.mat
        X = TrainSignal;
        ds_name = 'DoubleHeat';
        deg = 15;
        param.S = 2;  % number of subdictionaries 
    case 4
        load DataSetHeat30.mat
        load ComparisonHeat30.mat
        X = TrainSignal;
        ds_name = 'Heat';
        deg = 15;
        param.S = 1;  % number of subdictionaries 
end     

path = ['C:\Users\Cristina\Documents\GitHub\GraphLearningSparsityPriors\Results\05.07.18\',num2str(ds_name)];
param.N = size(X,1); % number of nodes in the graph
param.K = deg*ones(1,param.S);
param.J = param.N * param.S; % total number of atoms 
param.c = 1; % spectral control parameters
param.epsilon = 0.05;%0.02; % we assume that epsilon_1 = epsilon_2 = epsilon
param.mu = 1;%1e-2; % polynomial regularizer paremeter
param.y = X; %signals
param.y_size = size(param.y,2);

%% generate dictionary polynomial coefficients from heat kernel
% % % param.t(1) = 2; %heat kernel coefficients
% % % param.t(2) = 1; %this heat kernel will be inverted to cover high frequency components
% % % param.alpha = generate_coefficients(param);
K = max(param.K);
for i = 1:param.S
    param.alpha{i} = comp_alpha((K+1)*(i-1) + 1:(K+1)*i);
end
disp(param.alpha);


%% initialise learned data
param.T0 = 4; %sparsity level (# of atoms in each signals representation)
[param.Laplacian, learned_W] = init_by_weight(param.N);
[learned_dictionary, param] = construct_dict(param);
alpha = 5; %gradient descent parameter, it decreases with epochs
for big_epoch = 1:250
    param.testV = big_epoch;
    %% optimise with regard to x
    disp(['Epoch... ',num2str(big_epoch)]);
    x = OMP_non_normalized_atoms(learned_dictionary,param.y, param.T0);
    
    %% optimise with regard to W 
    maxEpoch = 1; %number of graph updating steps before updating sparse codes (x) again
    beta = 10^(-2); %graph sparsity penalty
    old_L = param.Laplacian;
    [param.Laplacian, learned_W] = update_graph_original(x, alpha, beta, maxEpoch, param,learned_W, learned_W);
    [learned_dictionary, param] = construct_dict(param);
    alpha = alpha*0.985; %gradient descent decreasing
end

%%
%constructed graph needs to be tresholded, otherwise it's too dense
%fix the number of desired edges here at nedges
nedges = length(find(comp_W))/2;
final_Laplacian = treshold_by_edge_number(param.Laplacian, nedges);
final_W = learned_W.*(final_Laplacian~=0);

CoefMatrix_Pol = OMP_non_normalized_atoms(learned_dictionary,TestSignal, param.T0);
errorTesting_Pol = sqrt(norm(TestSignal - learned_dictionary*CoefMatrix_Pol,'fro')^2/size(TestSignal,2));

tot_X = [x CoefMatrix_Pol];
norm_X = norm(tot_X);

%% Verify the results with the precision recall function
comp_L = diag(sum(comp_W,2)) - comp_W;
comp_Laplacian = (diag(sum(comp_W,2)))^(-1/2)*comp_L*(diag(sum(comp_W,2)))^(-1/2);

[optPrec, optRec, opt_Lapl] = precisionRecall(comp_Laplacian, param.Laplacian);

%% Save results
filename = [num2str(path),'\Output_Norm_PrecRec.mat'];
save(filename,'optPrec','optRec','CoefMatrix_Pol','errorTesting_Pol','norm_X');