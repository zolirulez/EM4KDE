clear all
close all

%% Load data
%load clusterdata2d % gives 'data' -- also try with other datasets!
load faithful
data = X;
[N,D] = size(data);
idx = randperm(N);
data = data(idx,:);
nfolds = 10;
data = data(1:N-rem(N,10),:); % To make data divisible by 10
[N,D] = size(data);
N_test = N/nfolds;
N_train = (nfolds-1) * N_test;

% Calculation of distances in advance
delta_data = repmat(data,1,N);
for it = 1:N
    delta_data(:,(it-1)*D+1:it*D) =...
        delta_data(:,(it-1)*D+1:it*D) - data(it,:);
end

%% Initialize parameters
K = N_train; % try with different parameters
pi_k = 1/K;
% Let mu_k be defined by kernel density estimation
%mu = data1;
% Let Sigma_k be the identity matrix:
Sigma = eye(D);

%% Loop until you're happy
max_iter = 500; % XXX: you should find a better convergence check than a max iteration counter
r = zeros(N_test,K);
Sigma = eye(D);
Sigmas = zeros(2,2,nfolds);
log_likelihood = zeros(nfolds, 1);
for iter = 1:max_iter
    fprintf('Iteration: %d\n', iter);
    %% Compute responsibilities
    for fold = 1:nfolds
        R = chol(Sigma,'upper');
        gain = 1/((2*pi)^(D/2)*det(R))/N_train;
        delta_x = [delta_data((fold-1)*N_test+1:fold*N_test,1:(fold-1)*N_test*D)...
            delta_data((fold-1)*N_test+1:fold*N_test,fold*N_test*D+1:end)];
        delta_x = reshape(delta_x,N_train*N_test,D);
        m = delta_x/R;
        r = reshape(gain*exp(-0.5*sum(m.*m,2)),N_test,N_train);
        rn = r./sum(r,2);
        %% Update parameters
        Sigmas(:,:,fold) = 1/N_test*((reshape(rn,N_train*N_test,1).*delta_x)'*delta_x);
        %% Compute log-likelihood of data
        R = chol(Sigma,'upper');
        gain = 1/((2*pi)^(D/2)*det(R))/N_train;
        delta_x = [delta_data((fold-1)*N_test+1:fold*N_test,1:(fold-1)*N_test*D)...
            delta_data((fold-1)*N_test+1:fold*N_test,fold*N_test*D+1:end)];
        delta_x = reshape(delta_x,N_train*N_test,D);
        m = delta_x/R;
        tmp_var = reshape(gain*exp(-0.5*sum(m.*m,2)),N_test,N_train);
        log_likelihood(fold) = sum(log(sum(tmp_var,2)),1);
    end
    % Extract the maximum likelihood sigma of the 10 folds
    [maxLL, maxIdx] = max(log_likelihood);
    log_lh(iter) = maxLL;
    
    % Put the MLE of sigma
    Sigma = Sigmas(:,:,maxIdx);
    log_likelihood = zeros(nfolds,1);
    % End...
    if iter > 1
        if abs(diff(log_lh(iter-1:iter)))<1e-5
            break;
        end
    end
end % for
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
for it = 1:N
    plot_normal(data(it,:), Sigma);
end % for
hold off
