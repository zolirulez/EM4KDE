% Normal dataset
close all
resolution = 10;
P = linspace(40,105,resolution)*1e5;
H = linspace(190,540,resolution)*1e3;
T = NaN(resolution);
T2 = NaN(resolution);
T5 = NaN(resolution);
normalstates = [];
faultystates2 = [];
faultystates5 = [];
for p = 1:length(P)
    for h = 1:length(H)
        T(h,p) = CoolProp.PropsSI('T','P',P(p),'H',H(h),'CO2');
        T(h,p) = sign(T(h,p))*min([abs(T(h,p)) 1e7]);
        normalstates = [normalstates; P(p),H(h),T(h,p)];
        if rem(p,2)
            if rem(h,2)
                T2(h,p) = T(h,p)+2; %2
                faultystates2 = [faultystates2; P(p),H(h),T2(h,p)];
            else
                T5(h,p) = T(h,p)-5;
                faultystates5 = [faultystates5; P(p),H(h),T5(h,p)];
            end
        else
            if rem(h,2)
                T2(h,p) = T(h,p)-2;%2
                faultystates2 = [faultystates2; P(p),H(h),T2(h,p)];
            else
                T5(h,p) = T(h,p)+5;
                faultystates5 = [faultystates5; P(p),H(h),T5(h,p)];
            end
        end
    end
end
% Saving data
save('phT_eval','normalstates','faultystates2','faultystates5')