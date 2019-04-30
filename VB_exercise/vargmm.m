clear all
close all

%% Load data
load clusterdata2d % gives 'data'
[N, D] = size(data);

%% Wrapper for digamma function
digamma = @(x) psi(0, x);

%% Number of components/clusters
K = 10;

%% Priors
a0 = 1e-3; % Mixing prior (small number: let the data speak)
m0 = zeros(1, D); B0 = 1e-3; % Gaussian mean prior
v0 = 3e1; W0 = eye(D)/v0; % Wishart covariance prior

%% Initialize parameters
m = cell(K, 1);
W = cell(K, 1);
B = repmat(B0 + N/K, [1, K]); % 1xK
a = repmat(a0 + N/K, [1, K]); % 1xK
v = repmat(v0 + N/K, [1, K]); % 1xK
for k = 1:K
    % Let m_k be a random data point:
    m{k} = data(randi(N), :); % 1xD
    % Let W_k be the mean of the Wishart prior:
    W{k} = v0*W0; % DxD % I MODIFIED HERE TODO
end % for
Nk = ones(1, K) / K; % 1xK

%% Loop until you're happy
max_iter = 100;
xb = cell(K,1);
S = cell(K,1);
lnrho = zeros(N,K);
r = zeros(N,K);
for iter = 1:max_iter
    %% Variational E-step
    % XXX: FILL ME IN!
    Elnpb = digamma(a) - digamma(sum(a));
    for k = 1:K
        ElnLt = sum(digamma((v(k)+1-(1:D))/2)) + D*log(2) + logdet(W{k});
        EmL = D/B(k) + v(k)*sum(((data - m{k})*W{k}).*(data - m{k}),2);
        lnrho(:,k) = Elnpb(k) + ElnLt/2 - D/2*log(2*pi) - 1/2*EmL;
    end
    lnC = -max(lnrho,[],2);
    r = exp(lnrho-lnC);
    r = r./sum(r,2);
    Nk = sum(r,1);
    
    
    %% Variational M-step
    % XXX: FILL ME IN!
    for k = 1:K
        rk = r(:,k)/Nk(k);
        rk(isnan(rk)) = 0;
        xb{k} = (rk'*data);
        S{k} = (rk.*(data - xb{k}))'*(data - xb{k});
        a(k) = a0 + Nk(k);
        B(k) = B0 + Nk(k);
        m{k} = (B0*m0 + Nk(k)*xb{k})/B(k);
        W{k} = pinv(inv(W0) + Nk(k)*S{k} + B0*Nk(k)/(B0 + Nk(k))*((xb{k} - m0)'*(xb{k} - m0)));
        v(k) = v0 + Nk(k);
    end
    v = v0 + Nk;
    
end % for

%% Plot data with distribution (we show expected distribution)
figure
plot(data(:, 1), data(:, 2), '.');
hold on
for k = 1:K
  if (Nk(k) > 0)
    plot_normal(m{k}, pinv(v(k) * (W{k})), 'linewidth', 2);
  end % if
end % for
hold off

%% Now, animate the uncertainty by sampling
%{
num_samples = 100;
figure
for s = 1:num_samples
  plot(data(:, 1), data(:, 2), '.');
  hold on
  for k = 1:K
    if (Nk(k) > 0)
      L = wishrnd(W_k{k}, v_k(k));
      Sigma = pinv(L);
      mu = mvnrnd(m_k{k}, Sigma/beta_k(k));
      plot_normal(mu, Sigma, 'linewidth', 2);
    end % if
  end % for
  hold off
  pause(0.1)
end % for

%% Animate the entire mixture distribution
figure
[X, Y] = meshgrid(linspace(-2, 2, 50));
XY = [X(:), Y(:)]; % 2500x2
for s = 1:num_samples
  pi_k = dirrnd(alpha_k);
  Z = zeros(size(X));
  for k = 1:K
    L = wishrnd(W_k{k}, v_k(k));
    Sigma = pinv(L);
    mu = mvnrnd(m_k{k}, Sigma/beta_k(k));
    Z(:) = Z(:) + pi_k(k) * mvnpdf(XY, mu, Sigma);
  end % for
  h = surfl(X, Y, Z); set(h, 'edgecolor', 'none');
  axis([-2, 2, -2, 2, 0, 0.3]);
  view(200*s/num_samples, 30);
  pause(0.1);
end % for
%}
