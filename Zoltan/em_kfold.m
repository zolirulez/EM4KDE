clear all
close all

%% Load data
load clusterdata2d % gives 'data' -- also try with other datasets!
% load faithful
% data = X;
[N,D] = size(data);
dataShuffled = data;
idx = randperm(N);
data = data(idx,:);
nfolds = 10;
data = data(1:N-rem(N,10),:); % To make data divisible by 10
[N,D] = size(data);
N_test = N/nfolds;
N_train = (nfolds-1) * N_test;
%% Initialize parameters
K = N_train; % try with different parameters
pi_k = 1/K;
% Let mu_k be defined by kernel density estimation
%mu = data1;
% Let Sigma_k be the identity matrix:
Sigma = eye(D);

%% Loop until you're happy
max_iter = 200; % XXX: you should find a better convergence check than a max iteration counter
r = zeros(N_test,K);
Sigma = eye(D);
Sigmas = zeros(2,2,nfolds);
log_likelihood = zeros(nfolds, 1);
CV = cvpartition(N, 'kfold',nfolds);
tic
for iter = 1:max_iter
    fprintf('Iteration: %d\n', iter);
    %% Compute responsibilities
    for fold = 1:nfolds
        R = chol(Sigma,'upper');
        gain = 1/((2*pi)^(D/2)*det(R))/N_train;
        mu = data(CV.training(fold),:);
        data2 = data(CV.test(fold),:);
        for k = 1:K
            m = (data2 - mu(k,:))/R;
            r(:,k) = gain*exp(-0.5*sum(m.*m,2));
        end
        rn = r./sum(r,2);
        %% Update parameters
        Sigma_sum = zeros(2,2);
        for k = 1:K
            Sigma_sum = Sigma_sum + ((rn(:,k).*(data2-mu(k,:)))'*(data2-mu(k,:)));
        end
        Sigmas(:,:,fold) = 1/N_test*Sigma_sum;
        
        %% Compute log-likelihood of data
        %log_likelihood(fold) = sum(log(sum(r,2)),1);
        tmp_var = zeros(N_test, N_train);
        for k = 1:K
            tmp_var(:,k) = tmp_var(:,k)  + mvnpdf(data2, mu(k,:), Sigmas(:,:,fold)) * pi_k;
        end
        log_likelihood(fold) = sum(log(sum(tmp_var,2)),1);
    end
    % Extract the maximum likelihood sigma of the 10 folds
    [maxLL, maxIdx] = max(log_likelihood);
    log_lh(iter) = maxLL;
    
    % Put the MLE of sigma
    Sigma = Sigmas(:,:,maxIdx);
    log_likelihood = zeros(nfolds,1);
    % End...
%     if iter > 1
%         if abs(diff(log_lh(iter-1:iter)))<1e-5
%             break;
%         end
%     end
end % for
toc
%% Plot log-likelihood -- did we converge?
figure(1)
plot(log_lh);
xlabel('Iterations'); ylabel('Log-likelihood');
title('Log-likelihood of each fold')

%% Plot data
figure(2);
if (D == 2)
    plot(data(:, 1), data(:, 2), '.');
elseif (D == 3)
    plot3(data(:, 1), data(:, 2), data(:, 3), '.');
end % if
hold on
for k = 1:K
    plot_normal(mu(k,:), Sigma);
end % for
hold off
