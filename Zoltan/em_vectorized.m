clear all
close all

%% Load data
faultswitch = 0;
dataswitch = 1;
if dataswitch
    %     load clusterdata2d
    load phT
    if faultswitch
        data = faultystates;
    else
        data = normalstates;
    end
    %     data = (data - mean(data))/cov(data);
else
    load faithful
    data = X;
%     data = (data - mean(data))/cov(data);
end
[N,D] = size(data);
idx = randperm(N);
data = data(idx,:);
data = data(1:N-rem(N,10),:); % To make data divisible by folds
if dataswitch
    nfolds = 10;
else
    nfolds = 5;
end
[N,D] = size(data);
N_test = N/nfolds;
N_train = (nfolds-1) * N_test;



%% Initialize parameters
K = N_train; % try with different parameters
pi_k = 1/K;
% Let mu_k be defined by kernel density estimation
%mu = data1;
% Let Sigma_k be the identity matrix:
initializations = 10;
Sigma_init = cell(initializations,1);
data_init = cell(initializations,1);
for init = 1:initializations
    data_init{init} = data;
    idx = randperm(N);
    data_init{init} = data_init{init}(idx,:);
    Sigma_init{init} = cov(data)*(eye(D) + 5*randn(D,D));
    Sigma_init{init} = Sigma_init{init} * Sigma_init{init}';
end
max_iter = 1000; 
% Log likelihood
LL = NaN(max_iter,initializations);

%% Loop until you're happy
% r = zeros(N_test,K);
% Sigma = eye(D);
% Sigmas = zeros(2,2,nfolds);
% log_likelihood = zeros(nfolds, 1);
% CV = cvpartition(N, 'kfold',nfolds);
% tic
% for iter = 1:max_iter
%     fprintf('Iteration: %d\n', iter);
%     %% Compute responsibilities
%     for fold = 1:nfolds
%         R = chol(Sigma,'upper');
%         gain = 1/((2*pi)^(D/2)*det(R))/N_train;
%         mu = data(CV.training(fold),:);
%         data2 = data(CV.test(fold),:);
%         for k = 1:K
%             m = (data2 - mu(k,:))/R;
%             r(:,k) = gain*exp(-0.5*sum(m.*m,2));
%         end
%         rn = r./sum(r,2);
%         %% Update parameters
%         Sigma_sum = zeros(2,2);
%         for k = 1:K
%             Sigma_sum = Sigma_sum + ((rn(:,k).*(data2-mu(k,:)))'*(data2-mu(k,:)));
%         end
%         Sigmas(:,:,fold) = 1/N_test*Sigma_sum;
%         
%         %% Compute log-likelihood of data
%         %log_likelihood(fold) = sum(log(sum(r,2)),1);
%         tmp_var = zeros(N_test, N_train);
%         for k = 1:K
%             tmp_var(:,k) = tmp_var(:,k)  + mvnpdf(data2, mu(k,:), Sigmas(:,:,fold)) * pi_k;
%         end
%         log_likelihood(fold) = sum(log(sum(tmp_var,2)),1);
%     end
%     % Extract the maximum likelihood sigma of the 10 folds
%     [maxLL, maxIdx] = max(log_likelihood);
%     LL(iter) = maxLL;
%     
%     % Put the MLE of sigma
%     Sigma = Sigmas(:,:,maxIdx);
%     log_likelihood = zeros(nfolds,1);
%     % End...
%     if iter > 5
%         if norm(pdist(LL(iter-4:iter)'),'inf') < 1e-12 % diff is faster!!
%             break;
%         end
%     end
% end % for
% toc
% Sigma
r_vector = zeros(N_test,K,nfolds);
Sigmas = zeros(D,D,nfolds);
log_likelihood = zeros(nfolds, 1);
maxIdx = 1;
tic
% Preparation of data
delta_x_ver3 = cell(nfolds,1);
for init = 1:initializations
    fprintf('Initialization iteration: %d\n', init);
    % Start from different initializations
    Sigma = Sigma_init{init};
    data = data_init{init};
    R = chol(Sigma,'upper');
    gain = 1/((2*pi)^(D/2)*det(R))/N_train;
    % Calculation of distances in advance
    delta_data = repmat(data,1,N);
    for it = 1:N
        delta_data(:,(it-1)*D+1:it*D) =...
            delta_data(:,(it-1)*D+1:it*D) - data(it,:);
    end
    for fold = 1:nfolds
        delta_x_ver1 = [delta_data((fold-1)*N_test+1:fold*N_test,1:(fold-1)*N_test*D)...
            delta_data((fold-1)*N_test+1:fold*N_test,fold*N_test*D+1:end)];
        delta_x_ver2 = zeros(size(delta_x_ver1));
        for iter2 = 1:D
            delta_x_ver2(:,(iter2-1)*N_train+1:iter2*N_train) = delta_x_ver1(:,iter2:D:end);
        end
        delta_x_ver3{fold} = reshape(delta_x_ver2,N_train*N_test,D);
        m = delta_x_ver3{fold}/R;
        r_vector(:,:,fold) = reshape(gain*exp(-0.5*sum(m.*m,2)),N_test,N_train);
    end
    for iter = 1:max_iter
        if rem(iter,100) == 0
            fprintf('Iteration: %d\n', iter);
        end
        %% Compute responsibilities
        for fold = 1:nfolds
            rn = r_vector(:,:,maxIdx)./sum(r_vector(:,:,maxIdx),2);
            %% Update parameters
            Sigmas(:,:,fold) = 1/N_test*((reshape(rn,N_train*N_test,1).*delta_x_ver3{fold})'*delta_x_ver3{fold});
            %% Compute log-likelihood of data
            R = chol(Sigmas(:,:,fold),'upper');
            gain = 1/((2*pi)^(D/2)*det(R))/N_train;
            m = delta_x_ver3{fold}/R;
            r = reshape(-0.5*sum(m.*m,2),N_test,N_train);
            lnC = -max(r); % This is not the recommended way
            r_vector(:,:,fold) = gain*exp(r + lnC);
            log_likelihood(fold) = sum(log(sum(r_vector(:,:,fold),2)),1);
        end
        % Extract the maximum likelihood sigma of the 10 folds
        [maxLL, maxIdx] = max(log_likelihood);
        LL(iter,init) = maxLL;
        
        % Put the MLE of sigma
        Sigma = Sigmas(:,:,maxIdx);
        % End...
        if iter > 5
            if norm(pdist(LL(iter-4:iter,init)),'inf') < 1e-12 % diff is faster!!
                break;
            end
        end
    end % for EM iterations
    LL(end,init) = maxLL;
    Sigma_init{init} = Sigma;
end % for initilizations
[maxLL, maxIdx] = max(LL(end,:));
Sigma = Sigma_init{maxIdx};
toc
%% Plot log-likelihood -- did we converge?
figure(1)
plot(LL);
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

%% Plot distribution
if (D ==2)
    figure(3)
    gm = gmdistribution(data,Sigma);
    if dataswitch
        fsurf(@(x,y)reshape(pdf(gm,[x(:),y(:)]),size(x)),[-2 2 -2 2])
    else
        fsurf(@(x,y)reshape(pdf(gm,[x(:),y(:)]),size(x)),[-1 5.5 -20 100])
    end
end
