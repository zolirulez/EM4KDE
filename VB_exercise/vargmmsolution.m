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
alpha0 = 1e-3; % Mixing prior (small number: let the data speak)
m0 = zeros(1, D); beta0 = 1e-3; % Gaussian mean prior
v0 = 3e1; W0 = eye(D)/v0; % Wishart covariance prior

%% Initialize parameters
m_k = cell(K, 1);
W_k = cell(K, 1);
beta_k = repmat(beta0 + N/K, [1, K]);
alpha_k = repmat(alpha0 + N/K, [1, K]);
v_k = repmat(v0 + N/K, [1, K]);
for k = 1:K
  % Let m_k be a random data point:
  m_k{k} = data(randi(N), :);
  % Let W_k be the mean of the Wishart prior:
  W_k{k} = v0*W0;
end % for

%% Loop until you're happy
max_iter = 100;
for iter = 1:max_iter
    %% Variational E-step
    ln_rho = NaN(N, K); % move out of loop?
    Elnpi = digamma(alpha_k) - digamma(sum(alpha_k));
    for k = 1:K
        delta_k = bsxfun(@minus, data, m_k{k}); % NxD
        EmuL = D/beta_k(k) + v_k(k) * sum((delta_k * W_k{k}) .* delta_k, 2); % Nx1
        ElnL = sum(digamma(0.5*(v_k(k) + 1 - (1:D)))) + D*log(2) + logdet(W_k{k}); % 1x1
        ln_rho(:, k) = Elnpi(k) + 0.5*ElnL - 0.5*D*log(2*pi) - 0.5 * EmuL; % Nx1
    end % for
    rho = exp(bsxfun(@minus, ln_rho, max(ln_rho, [], 2))); % NxK
    r_nk = bsxfun(@times, rho, 1./sum(rho, 2)); % NxK
    
    %% Variational M-step
    Nk = sum(r_nk, 1); % 1xK
    alpha_k = alpha0 + Nk; % 1xK
    beta_k = beta0 + Nk; % 1xK
    v_k = v0 + Nk; % 1xK
    for k = 1:K
        rk = r_nk(:, k) / Nk(k); % Nx1
        rk(isnan(rk)) = 0;
        xbar = rk.' * data; % 1xD
        delta_k = bsxfun(@minus, data, xbar); % NxD
        Sk = delta_k.' * spdiags(rk, 0, N, N) * delta_k; % DxD
        
        m_k{k} = (beta0*m0 + Nk(k)*xbar) / beta_k(k);
        Winv = inv(W0) + Nk(k) * Sk + (beta0*Nk(k)/(beta0 + Nk(k)))*((xbar - m0).' * (xbar - m0));
        W_k{k} = pinv(Winv);
    end % for
end % for

%% Plot data with distribution (we show expected distribution)
figure
plot(data(:, 1), data(:, 2), '.');
hold on
for k = 1:K
  if (Nk(k) > 0)
    plot_normal(m_k{k}, pinv(v_k(k) * (W_k{k})), 'linewidth', 2);
  end % if
end % for
hold off

% %% Now, animate the uncertainty by sampling
% % {
% num_samples = 100;
% figure
% pause()
% for s = 1:num_samples
%   plot(data(:, 1), data(:, 2), '.');
%   hold on
%   for k = 1:K
%     if (Nk(k) > 0)
%       L = wishrnd(W_k{k}, v_k(k));
%       Sigma = pinv(L);
%       mu = mvnrnd(m_k{k}, Sigma/beta_k(k));
%       plot_normal(mu, Sigma, 'linewidth', 2);
%     end % if
%   end % for
%   hold off
%   pause(0.1)
% end % for
% 
% %% Animate the entire mixture distribution
% figure
% pause()
% [X, Y] = meshgrid(linspace(-2, 2, 50));
% XY = [X(:), Y(:)]; % 2500x2
% for s = 1:num_samples
%   pi_k = dirrnd(alpha_k);
%   Z = zeros(size(X));
%   for k = 1:K
%     L = wishrnd(W_k{k}, v_k(k));
%     Sigma = pinv(L);
%     mu = mvnrnd(m_k{k}, Sigma/beta_k(k));
%     Z(:) = Z(:) + pi_k(k) * mvnpdf(XY, mu, Sigma);
%   end % for
%   h = surfl(X, Y, Z); set(h, 'edgecolor', 'none');
%   axis([-2, 2, -2, 2, 0, 0.3]);
%   view(200*s/num_samples, 30);
%   pause(0.1);
% end % for
% %}