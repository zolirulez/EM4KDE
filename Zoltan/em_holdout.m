clear all
close all

%% Load data
load clusterdata2d % gives 'data' -- also try with other datasets!
% load faithful
% data = X;
rowSize = size(data,1);
dataShuffled = data;
idx = randperm(rowSize);
data = data(idx,:);
cut = 1000;
data1 = data(1:cut,:);
data2 = data(cut+1:end,:);
[Ntrain, D] = size(data1);
[Ntest, ~] = size(data2);

%% Initialize parameters
K = Ntrain; % try with different parameters
mu = cell(K, 1);
Sigma = cell(K, 1);
pi_k = ones(K, 1)/K;
% Let mu_k be defined by kernel density estimation
mu = data1;
% Let Sigma_k be the identity matrix:
Sigma = eye(D);

%% Loop until you're happy
max_iter = 1000; % XXX: you should find a better convergence check than a max iteration counter
log_likelihood = NaN(max_iter, 1);
r = zeros(Ntest,K);
tic
for iter = 1:max_iter
    %% Compute responsibilities
    for k = 1:K
        r(:,k) = pi_k(k)*mvnpdf(data2, mu(k,:), Sigma);
    end
    rn = r./sum(r,2);
    Nk = sum(rn,1);
    %% Update parameters
    Sigma_new = zeros(2,2);
    for k = 1:K
        Sigma_new = Sigma_new + 1/Ntest*((rn(:,k).*(data2-mu(k,:)))'*(data2-mu(k,:)));
    end
    Sigma = Sigma_new;
    
    %% Compute log-likelihood of data
    for rowSize = Ntest
        log_likelihood(iter) = sum(log(sum(r,2)),1);
    end
    
    % End...
    if iter > 1
        if abs(diff(log_likelihood(iter-1:iter)))<1e-5
            break;
        end
    end
end % for
toc
Sigma
% Let Sigma_k be the identity matrix:
Sigma = eye(D);
tic
for iter = 1:max_iter
    %% Compute responsibilities
    R = chol(Sigma,'upper');
    gain = 1/((2*pi)^(D/2)*det(R))/Ntrain;
    for k = 1:K
        m = (data2 - mu(k,:))/R;
        r(:,k) = gain*exp(-0.5*sum(m.*m,2));
    end
    rn = r./sum(r,2);
    Nk = sum(rn,1);
    %% Update parameters
    Sigma_new = zeros(2,2);
    for k = 1:K
        Sigma_new = Sigma_new + 1/Ntest*((rn(:,k).*(data2-mu(k,:)))'*(data2-mu(k,:)));
    end
    Sigma = Sigma_new;
    
    %% Compute log-likelihood of data
    for rowSize = Ntest
        log_likelihood(iter) = sum(log(sum(r,2)),1);
    end
    
    % End...
    if iter > 1
        if abs(diff(log_likelihood(iter-1:iter)))<1e-5
            break;
        end
    end
end % for
toc
Sigma
%% Plot log-likelihood -- did we converge?
figure
plot(log_likelihood);
xlabel('Iterations'); ylabel('Log-likelihood');

%% Plot data
figure
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
