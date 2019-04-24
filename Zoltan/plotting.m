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
    switch dataswitch
        case 2
            fsurf(@(x,y)reshape(pdf(gm,[x(:),y(:)]),size(x)),[-2 2 -2 2])
        case 1
            fsurf(@(x,y)reshape(pdf(gm,[x(:),y(:)]),size(x)),[-1 5.5 -20 100])
    end
end