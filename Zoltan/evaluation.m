figure(1)
[X,Y,T,AUC] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [fault_evaluation(1:N_eval/2); fault_evaluation(N_eval/2+1:N_eval/4*3)],1);
plot(X,Y)
hold on
plot([0 1],[0 Y(end)])
hold off
title(['ROC for 2 degrees deviation, AUC: ' num2str(AUC)])
figure(2)
[X,Y,T,AUC] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [fault_evaluation(1:N_eval/2); fault_evaluation(N_eval/4*3+1:end)],1);
plot(X,Y)
hold on
plot([0 1],[0 Y(end)])
hold off
title(['ROC for 5 degrees deviation, AUC: ' num2str(AUC)])
figure(3)
[X,Y,T,AUC] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [fault_evaluation_noisy(1:N_eval/2); fault_evaluation_noisy(N_eval/2+1:N_eval/4*3)],1);
plot(X,Y)
hold on
plot([0 1],[0 Y(end)])
hold off
title(['ROC for 2 degrees deviation,noisy, AUC: ' num2str(AUC)])
figure(4)
[X,Y,T,AUC] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [fault_evaluation_noisy(1:N_eval/2); fault_evaluation_noisy(N_eval/4*3+1:end)],1);
plot(X,Y)
hold on
plot([0 1],[0 Y(end)])
hold off
title(['ROC for 5 degrees deviation,noisy, AUC: ' num2str(AUC)])

% Finding optimal threshold
figure(5)
plot(T,Y./X)
hold on
plot(T,X./(1-Y))
hold off
xlabel('probability threshold')
% Y: TPR, X: FPR
legend('Detection/missed detection','Missed detection/false alarm')
% TODO

% Comparison to logistic regression
figure(6)
logreg = fitglm(data(:,1:2),data(:,3),'Distribution','normal','Link','log');
[X,Y,T,AUC] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [ones(N_eval*3/4,1) [evaldata(1:N_eval/2,1:2); evaldata(N_eval/4*3+1:end,1:2)]]*logreg.Coefficients{:,1},1);
plot(X,Y)
hold on
plot([0 1],[0 Y(end)])
hold off
title(['LOGREG, ROC for 5 degrees deviation,noisy, AUC: ' num2str(AUC)])