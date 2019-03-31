clear all
%close all

%% Load data
load clusterdata2d % gives 'data' -- also try with other datasets!
% load faithful
% data = X;
[N,D] = size(data);
dataShuffled = data;
idx = randperm(N);
data = data(idx,:);
data = data(1:end-3,:); % To make data divisible by 10
nfolds = 10;
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
max_iter = 1000; % XXX: you should find a better convergence check than a max iteration counter
r = zeros(N_test,K);
Sigmas = cell(nfolds,1);

tic
CV = cvpartition(N,'kfold',nfolds);
for fold = 1:nfolds
    fprintf('Computing fold: %d/%d \n', fold, nfolds); 
    % Let Sigma_k be the identity matrix for every fold:
    Sigma = eye(D);
    R = chol(Sigma,'upper');
    gain = 1/((2*pi)^(D/2)*det(R))/N_train;
    for iter = 1:max_iter
        %% Compute responsibilities
        mu = data(CV.training(fold),:);
        data2 = data(CV.test(fold),:);
        for k = 1:K
            m = (data2 - mu(k,:))/R;
            r(:,k) = gain*exp(-0.5*sum(m.*m,2));
        end
        rn = r./sum(r,2);
        Nk = sum(rn,1);
        %% Update parameters
        Sigma_sum = zeros(2,2);
        for k = 1:K
            Sigma_sum = Sigma_sum + ((rn(:,k).*(data2-mu(k,:)))'*(data2-mu(k,:)));
        end
        Sigma = 1/N_test*Sigma_sum;

        %% Compute log-likelihood of data
        for N = N_test
            log_likelihood(fold,iter) = sum(log(sum(r,2)),1);
        end

        % End...
        if iter > 1
            if abs(diff(log_likelihood(iter-1:iter)))<1e-5
                break;
            end
        end
    end % for
    toc
    Sigma;
    Sigmas{fold} = Sigma;
    fold_likelihood(fold) = log_likelihood(fold, end);
end
%% Plot log-likelihood -- did we converge?
figure('fold-loglike')
plot(fold_likelihood);
xlabel('Iterations'); ylabel('Log-likelihood');
title('Log-likelihood of each fold')

%% Plot data
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
