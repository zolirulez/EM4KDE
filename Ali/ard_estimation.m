
% Load data 
addpath('C:\Users\u375749\Documents\Semester4temporary\EM4KDE\Zoltan')

load phT.mat
data = normalstates;

%load faithful.mat
%data = X;

stand_data = (data - mean(data)) ./ std(data);
%for k = 1:300
% Number of neighbors
K = 20;

% Find the k nearest neighbors of your data
% Remember to define K as K+1 since we exclude the point itself from 
% being a neighbor
[i,Dist] = knnsearch(stand_data, stand_data, 'K', K+1);


% Compute the ard for the observations already existing in
% the dataset
densityX = 1./mean(Dist(:,2:end),2);
ard_X = densityX ./ mean(densityX(i(:, 2:end)), 2);

%% EVALUATION
% Make a test set which contains outliers and normal data points with very
% little noise
%load phT_eval_forkde.mat
load phT_eval_forkde.mat
n = length(faultystates20);
outlier_data = faultystates20;
%outliers = (outlier_data - mean(outlier_data)) ./ std(outlier_data);
%new_data = (normalstates - mean(normalstates) ) ./ std(normalstates);
new_data = normalstates;
% Concatenate the data
test_set = [outlier_data; new_data];
% test_set(:,3) = test_set(:,3) + rand(length(test_set),1)*1-1;
test_set = (test_set - mean(test_set) ) ./ std(test_set);
labels = [ones(n,1); zeros(length(new_data),1)];

% Shuffle the data up
idx = randperm(length(labels));
test_set = test_set(idx, :);
labels = labels(idx);

% Find the k nearest neighbors in test set
[i_t,Dist_t] = knnsearch(stand_data, test_set, 'K', 30);

% Compute the density of the test points
density_t = 1./ mean(Dist_t,2);

% Compute the ard of the test points
ard_t = density_t ./ mean(densityX(i_t(:,2:end)), 2);

[FPR20_ARD,TPR20_ARD,T,AUC20_ARD] = perfcurve(labels, ard_t, 0);
%auc_arr(k) = AUC;
AUC20_ARD
% mfig('ROC');
hold on
plot(FPR20_ARD,TPR20_ARD);
hold off
xlabel('False positive rate') 
ylabel('True positive rate')
save('OD','FPR20_ARD','TPR20_ARD','AUC20_ARD','-append')
%%
%mfig('AUC');
%plot(auc_arr);
%xlabel('K in KNN');
%ylabel('AUC');





