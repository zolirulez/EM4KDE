figure(1)
[FPR2_EMKDE,TPR2_EMKDE,T,AUC2_EMKDE] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [fault_evaluation(1:N_eval/2); fault_evaluation(N_eval/2+1:N_eval/4*3)],1);
plot(FPR2_EMKDE,TPR2_EMKDE)
hold on
plot([0 1],[0 1])
hold off
title(['ROC for 2 degrees deviation, AUC: ' num2str(AUC2_EMKDE)])
save('CC','FPR2_EMKDE','TPR2_EMKDE','AUC2_EMKDE','-append')
figure(2)
[FPR5_EMKDE,TPR5_EMKDE,T,AUC5_EMKDE] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [fault_evaluation(1:N_eval/2); fault_evaluation(N_eval/4*3+1:end)],1);
plot(FPR5_EMKDE,TPR5_EMKDE)
hold on
plot([0 1],[0 1])
hold off
title(['ROC for 5 degrees deviation, AUC: ' num2str(AUC5_EMKDE)])
save('CC','FPR5_EMKDE','TPR5_EMKDE','AUC5_EMKDE','-append')
figure(3)
[FPR2N_EMKDE,TPR2N_EMKDE,T,AUC2N_EMKDE] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [fault_evaluation_noisy(1:N_eval/2); fault_evaluation_noisy(N_eval/2+1:N_eval/4*3)],1);
plot(FPR2N_EMKDE,TPR2N_EMKDE)
hold on
plot([0 1],[0 1])
hold off
title(['ROC for 2 degrees deviation,noisy, AUC: ' num2str(AUC2N_EMKDE)])
save('CC','FPR2N_EMKDE','TPR2N_EMKDE','AUC2N_EMKDE','-append')
figure(4)
[FPR5N_EMKDE,TPR5N_EMKDE,T,AUC5N_EMKDE] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [fault_evaluation_noisy(1:N_eval/2); fault_evaluation_noisy(N_eval/4*3+1:end)],1);
plot(FPR5N_EMKDE,TPR5N_EMKDE)
hold on
plot([0 1],[0 1])
hold off
title(['ROC for 5 degrees deviation,noisy, AUC: ' num2str(AUC5N_EMKDE)])
save('CC','FPR5N_EMKDE','TPR5N_EMKDE','AUC5N_EMKDE','-append')

% Finding optimal threshold
figure(5)
plot(T,1-TPR5N_EMKDE)
hold on
plot(T,FPR5N_EMKDE)
hold off
xlabel('probability threshold')
% Y: TPR, X: FPR
legend('Missed detection','False alarm')

% Comparison to logistic regression
load phT_eval
evaldata = [normalstates; faultystates2; faultystates5];
figure(6)
logreg = fitglm(data(:,1:2),data(:,3),'Distribution','normal','Link','log');
[FPR5_LR,TPR5_LR,~,AUC5_LR] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [ones(N_eval*3/4,1) [evaldata(1:N_eval/2,1:2); evaldata(N_eval/4*3+1:end,1:2)]]*logreg.Coefficients{:,1},1);
plot(FPR5_LR,TPR5_LR)
hold on
plot([0 1],[0 1])
hold off
title(['LOGREG, ROC for 5 degrees deviation, AUC: ' num2str(AUC5_LR)])
save('CC','FPR5_LR','TPR5_LR','AUC5_LR','-append')
figure(7)
[FPR2_LR,TPR2_LR,~,AUC2_LR] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [ones(N_eval*3/4,1) [evaldata(1:N_eval/2,1:2); evaldata(N_eval/2+1:N_eval/4*3,1:2)]]*logreg.Coefficients{:,1},1);
plot(FPR2_LR,TPR2_LR)
hold on
plot([0 1],[0 1])
hold off
title(['LOGREG, ROC for 2 degrees deviation, AUC: ' num2str(AUC2_LR)])
save('CC','FPR2_LR','TPR2_LR','AUC2_LR','-append')

evaldata_noisy = evaldata;
evaldata_noisy(:,3) = evaldata_noisy(:,3) + rand(length(evaldata),1)*1-1;
figure(8)
logreg = fitglm(data(:,1:2),data(:,3),'Distribution','normal','Link','log');
[FPR5N_LR,TPR5N_LR,~,AUC5N_LR] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [ones(N_eval*3/4,1) [evaldata_noisy(1:N_eval/2,1:2); evaldata_noisy(N_eval/4*3+1:end,1:2)]]*logreg.Coefficients{:,1},1);
plot(FPR5N_LR,TPR5N_LR)
hold on
plot([0 1],[0 1])
hold off
title(['LOGREG, ROC for 5 degrees deviation,noisy, AUC: ' num2str(AUC5N_LR)])
save('CC','FPR5N_LR','TPR5N_LR','AUC5N_LR','-append')
figure(9)
[FPR2N_LR,TPR2N_LR,~,AUC2N_LR] = perfcurve([zeros(N_eval/2,1); ones(N_eval/4,1)],...
    [ones(N_eval*3/4,1) [evaldata_noisy(1:N_eval/2,1:2); evaldata_noisy(N_eval/2+1:N_eval/4*3,1:2)]]*logreg.Coefficients{:,1},1);
plot(FPR2N_LR,TPR2N_LR)
hold on
plot([0 1],[0 1])
hold off
title(['LOGREG, ROC for 2 degrees deviation,noisy, AUC: ' num2str(AUC2N_LR)])
save('CC','FPR2N_LR','TPR2N_LR','AUC2N_LR','-append')