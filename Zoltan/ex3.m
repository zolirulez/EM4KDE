clear all
close all

%% Load data
load clusterdata2d % gives 'data' -- also try with other datasets!
[N, D] = size(data);

%% Initialize parameters
K = 3; % try with different parameters
mu = cell(K, 1);
Sigma = cell(K, 1);
pi_k = ones(K, 1)/K;
for k = 1:K
  % Let mu_k be a random data point:
  mu{k} = data(randi(N), :);
  % Let Sigma_k be the identity matrix:
  Sigma{k} = eye(D);
end % for

%% Loop until you're happy
max_iter = 1000; % XXX: you should find a better convergence check than a max iteration counter
log_likelihood = NaN(max_iter, 1);
r = zeros(N,K);
for iter = 1:max_iter
  %% Compute responsibilities
  % XXX: FILL ME IN!
  for k = 1:K
      r(:,k) = pi_k(k)*mvnpdf(data, mu{k}, Sigma{k});
  end
  rn = r./sum(r,2);
  %% Update parameters
  Nk = sum(rn,1);
  % XXX: FILL ME IN!
  mu_new = cell(K,1);
  Sigma_new = cell(K,1);
  mu_new(:) = {zeros(1,2)};
  Sigma_new(:) = {zeros(2,2)};
  for k = 1:K
      for n = 1:N
          mu_new{k} = mu_new{k} + 1/Nk(k)*rn(n,k)*data(n,:);
      end
      for n = 1:N
          Sigma_new{k} = Sigma_new{k} + 1/Nk(k)*rn(n,k)*((data(n,:)-mu_new{k})'*(data(n,:)-mu_new{k}));
      end
  end
  pi_k_new = Nk'/N;
  mu = mu_new;
  Sigma = Sigma_new;
  pi_k = pi_k_new;
  
  %% Compute log-likelihood of data
  % log_likelihood(iter) = NaN; % XXX: DO THIS CORRECTLY
  for n = N
      log_likelihood(iter) = sum(log(sum(r,2)),1);
  end
  
  % End...
  if iter > 1
      if abs(diff(log_likelihood(iter-1:iter)))<1e-5
          break;
      end
  end
end % for

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
  plot_normal(mu{k}, Sigma{k});
end % for
hold off
