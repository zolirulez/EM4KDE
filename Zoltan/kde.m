clearvars
dataswitch = 0;
faultswitch = 0;
%% Load data
switch dataswitch
    case 0
        load phT
        if faultswitch
            data = faultystates;
        else
            data = normalstates;
        end
    case 1
        load faithful
        data = X;
    case 2
        load clusterdata2d
end
data = (data - mean(data))/cov(data);
[N,D] = size(data);
idx = randperm(N);
data = data(idx,:);
data = data(1:N-rem(N,10),:); % To make data divisible by folds
switch dataswitch
    case 0
        nfolds = 10;
    case 1
        nfolds = 5;
    case 2
        nfolds = 10;
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
end

%% Loop until you're happy
r_vector = zeros(N_test,K,nfolds);
nSfolds = 50;
lambda = logspace(-15,0,nSfolds);
Sigmas = zeros(D,D,nSfolds);
for Sfolds = 1:nSfolds
    Sigmas(:,:,Sfolds) = lambda(Sfolds)*eye(D);
end
log_likelihood = zeros(Sfolds, 1);
maxIdx = 1;
tic
maxperset = NaN(initializations,nfolds,2);
% Preparation of data
delta_x_ver3 = cell(nfolds,1);
for init = 1:initializations
    fprintf('Initialization iteration: %d\n', init);
    % Start from different initializations
    data = data_init{init};
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
    end
    for fold = 1:nfolds
        % Compute responsibilities
        for Sfold = 1:nSfolds
            % Compute log-likelihood of data
            R = chol(Sigmas(:,:,Sfold),'upper');
            gain = 1/((2*pi)^(D/2)*det(R))/N_train;
            m = delta_x_ver3{fold}/R;
            r = reshape(-0.5*sum(m.*m,2),N_test,N_train);
            lnC = -max(r); % This is not the recommended way
            r_vector = gain*exp(r + lnC*0);
            log_likelihood(Sfold) = sum(log(sum(r_vector,2)),1);
        end
        % Extract the maximum likelihood sigma of the 10 Sigma folds
        [maxLL, maxIdx] = max(log_likelihood);
        maxperset(init,fold,1) = maxLL;
        maxperset(init,fold,2) = maxIdx;
    end
end % for initilizations
toc

figure(1)
imagesc(maxperset(:,:,1))
colormap(jet)
set(gca,'XTick',[],'YTick',[],'YDir','normal')
[x,y] = meshgrid(1:nfolds,1:initializations);
text(x(:),y(:),num2str(reshape(maxperset(:,:,2),initializations*nfolds,1)),'HorizontalAlignment','center')
robustlambdaidx = mean(mean(maxperset(:,:,2)))
Sigma = Sigmas(:,:,ceil(robustlambdaidx));
ylabel('Initializations')
xlabel('Data folds')

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
    switch dataswitch
        case 2
            fsurf(@(x,y)reshape(pdf(gm,[x(:),y(:)]),size(x)),[-2 2 -2 2])
        case 1
            fsurf(@(x,y)reshape(pdf(gm,[x(:),y(:)]),size(x)),[-1 5.5 -20 100])
    end
end

%% Density evaluation of evaluation data set
load phT_eval_forkde
evaldata = [normalstates; faultystates5; faultystates20];
evaldata = (evaldata - mean(evaldata))/cov(evaldata);
N = length(data);
N_eval = length(evaldata);
delta_data_ver1 = repmat(evaldata,1,N);
for it = 1:N
    delta_data_ver1(:,(it-1)*D+1:it*D) =...
        delta_data_ver1(:,(it-1)*D+1:it*D) - data(it,:);
end
delta_data_ver2 = zeros(size(delta_data_ver1));
for iter2 = 1:D
    delta_data_ver2(:,(iter2-1)*N+1:iter2*N) = delta_data_ver1(:,iter2:D:end);
end
delta_data_ver3 = reshape(delta_data_ver2,N*N_eval,D);
R = chol(Sigma,'upper');
gain = 1/((2*pi)^(D/2)*det(R))/N;
m = delta_data_ver3/R;
r = reshape(-0.5*sum(m.*m,2),N*N_eval,1);
lnC = -max(r); % This is not the recommended way
r_vector = gain*exp(r + lnC*0);
density = sum(reshape(r_vector,N_eval,N),2);
%% Evaulation
figure(3)
[X,Y,T,AUC] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [1-density(1:N_eval/2); 1-density(N_eval/2+1:N_eval/4*3)],1);
plot(X,Y)
hold on
plot([0 1],[0 Y(end)])
hold off
title(['KDE, ROC for 5 degrees deviation, AUC: ' num2str(AUC)])
figure(4)
[X,Y,T,AUC] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [1-density(1:N_eval/2); 1-density(N_eval/4*3+1:end)],1);
plot(X,Y)
hold on
plot([0 1],[0 Y(end)])
hold off
title(['KDE, ROC for 20 degrees deviation, AUC: ' num2str(AUC)])
%% Density evaluation of evaluation data set, EMKDE
load phT
data = normalstates;
load phT_eval
load SigmaNormal
evaldata = [normalstates; faultystates2; faultystates5];
N = length(data);
N_eval = length(evaldata);
delta_data_ver1 = repmat(evaldata,1,N);
for it = 1:N
    delta_data_ver1(:,(it-1)*D+1:it*D) =...
        delta_data_ver1(:,(it-1)*D+1:it*D) - data(it,:);
end
delta_data_ver2 = zeros(size(delta_data_ver1));
for iter2 = 1:D
    delta_data_ver2(:,(iter2-1)*N+1:iter2*N) = delta_data_ver1(:,iter2:D:end);
end
delta_data_ver3 = reshape(delta_data_ver2,N*N_eval,D);
R = chol(SigmaNormal,'upper');
gain = 1/((2*pi)^(D/2)*det(R))/N;
m = delta_data_ver3/R;
r = reshape(-0.5*sum(m.*m,2),N*N_eval,1);
lnC = -max(r); % This is not the recommended way
r_vector = gain*exp(r + lnC*0);
density = sum(reshape(r_vector,N_eval,N),2);
%% Evaulation, EMKDE
figure(5)
[X,Y,T,AUC] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [1-density(1:N_eval/2); 1-density(N_eval/2+1:N_eval/4*3)],1);
plot(X,Y)
hold on
plot([0 1],[0 Y(end)])
hold off
title(['EMKDE, ROC for 5 degrees deviation, AUC: ' num2str(AUC)])
figure(6)
[X,Y,T,AUC] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [1-density(1:N_eval/2); 1-density(N_eval/4*3+1:end)],1);
plot(X,Y)
hold on
plot([0 1],[0 Y(end)])
hold off
title(['EMKDE, ROC for 20 degrees deviation, AUC: ' num2str(AUC)])