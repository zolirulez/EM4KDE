% CLASSIFICATION
load CC

h = figure(11)
plot(FPR2_LR,TPR2_LR,'b-')
hold on
plot(FPR2_EMKDE,TPR2_EMKDE,'k-')
plot([0 1],[0 1],'r--')
hold off
legend(['Log. Reg., AUC: ' num2str(AUC2_LR)],...
    ['EM4KDE, AUC: ' num2str(AUC2_EMKDE)],'location','southeast')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
saveas(h,'CC2.png')

h = figure(12)
plot(FPR2N_LR,TPR2N_LR,'b-')
hold on
plot(FPR2N_EMKDE,TPR2N_EMKDE,'k-')
plot([0 1],[0 1],'r--')
hold off
legend(['Log. Reg., AUC: ' num2str(AUC2N_LR)],...
    ['EM4KDE, AUC: ' num2str(AUC2N_EMKDE)],'location','southeast')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
saveas(h,'CC2N.png')

h = figure(13)
plot(FPR5_LR,TPR5_LR,'b-')
hold on
plot(FPR5_EMKDE,TPR5_EMKDE,'k-')
plot([0 1],[0 1],'r--')
hold off
legend(['Log. Reg., AUC: ' num2str(AUC5_LR)],...
    ['EM4KDE, AUC: ' num2str(AUC5_EMKDE)],'location','southeast')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
saveas(h,'CC5.png')

h = figure(14)
plot(FPR5N_LR,TPR5N_LR,'b-')
hold on
plot(FPR5N_EMKDE,TPR5N_EMKDE,'k-')
plot([0 1],[0 1],'r--')
hold off
legend(['Log. reg., AUC: ' num2str(AUC5N_LR)],...
    ['EM4KDE, AUC: ' num2str(AUC5N_EMKDE)],'location','southeast')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
saveas(h,'CC5N.png')




% OUTLIER DETECTION
load OD

h = figure(21)
plot(FPR2_KDE,TPR2_KDE,'b-')
hold on
plot(FPR2_EMKDE,TPR2_EMKDE,'k-')
plot(FPR2_ARD,TPR2_ARD,'m-')
plot([0 1],[0 1],'r--')
hold off
legend(['Kernel Density Estimation, AUC: ' num2str(AUC2_KDE)],...
    ['EM4KDE, AUC: ' num2str(AUC2_EMKDE)],...
    ['Average Relative Density, AUC: ' num2str(AUC2_ARD)],'location','southeast')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
saveas(h,'OD2.png')

h = figure(22)
plot(FPR2N_KDE,TPR2N_KDE,'b-')
hold on
plot(FPR2N_EMKDE,TPR2N_EMKDE,'k-')
plot(FPR2N_ARD,TPR2N_ARD,'m-')
plot([0 1],[0 1],'r--')
hold off
legend(['Kernel Density Estimation, AUC: ' num2str(AUC2N_KDE)],...
    ['EM4KDE, AUC: ' num2str(AUC2N_EMKDE)],...
    ['Average Relative Density, AUC: ' num2str(AUC2N_ARD)],'location','southeast')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
saveas(h,'OD2N.png')

h = figure(23)
plot(FPR5_KDE,TPR5_KDE,'b-')
hold on
plot(FPR5_EMKDE,TPR5_EMKDE,'k-')
plot(FPR5_ARD,TPR5_ARD,'m-')
plot([0 1],[0 1],'r--')
hold off
legend(['Kernel Density Estimation, AUC: ' num2str(AUC5_KDE)],...
    ['EM4KDE, AUC: ' num2str(AUC5_EMKDE)],...
    ['Average Relative Density, AUC: ' num2str(AUC5_ARD)],'location','southeast')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
saveas(h,'OD5.png')

h = figure(24)
plot(FPR5N_KDE,TPR5N_KDE,'b-')
hold on
plot(FPR5N_EMKDE,TPR5N_EMKDE,'k-')
plot(FPR5N_ARD,TPR5N_ARD,'m-')
plot([0 1],[0 1],'r--')
hold off
legend(['Kernel Density Estimation, AUC: ' num2str(AUC5N_KDE)],...
    ['EM4KDE, AUC: ' num2str(AUC5N_EMKDE)],...
    ['Average Relative Density, AUC: ' num2str(AUC5N_ARD)],'location','southeast')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
saveas(h,'OD5N.png')

h = figure(25)
plot(FPR20_KDE,TPR20_KDE,'b-')
hold on
plot(FPR20_EMKDE,TPR20_EMKDE,'k-')
plot(FPR20_ARD,TPR20_ARD,'m-')
plot([0 1],[0 1],'r--')
hold off
legend(['Kernel Density Estimation, AUC: ' num2str(AUC20_KDE)],...
    ['EM4KDE, AUC: ' num2str(AUC20_EMKDE)],...
    ['Average Relative Density, AUC: ' num2str(AUC20_ARD)],'location','southeast')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
saveas(h,'OD20.png')

h = figure(26)
plot(FPR20N_KDE,TPR20N_KDE,'b-')
hold on
plot(FPR20N_EMKDE,TPR20N_EMKDE,'k-')
plot(FPR20N_ARD,TPR20N_ARD,'m-')
plot([0 1],[0 1],'r--')
hold off
legend(['Kernel Density Estimation, AUC: ' num2str(AUC20N_KDE)],...
    ['EM4KDE, AUC: ' num2str(AUC20N_EMKDE)],...
    ['Average Relative Density, AUC: ' num2str(AUC20N_ARD)],'location','southeast')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
saveas(h,'OD20N.png')