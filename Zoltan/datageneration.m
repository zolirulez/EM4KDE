% Normal dataset
close all
resolution = 25;
P = linspace(45,100,resolution)*1e5;
H = linspace(200,525,resolution)*1e3;
T = NaN(resolution);
Tf = NaN(resolution);
normalstates = [];
faultystates = [];
for p = 1:length(P)
    for h = 1:length(H)
        T(h,p) = CoolProp.PropsSI('T','P',P(p),'H',H(h),'CO2');
        T(h,p) = sign(T(h,p))*min([abs(T(h,p)) 1e7]);
        normalstates = [normalstates; P(p),H(h),T(h,p)];
        if rem(p,2)
            if rem(h,2)
                Tf(h,p) = T(h,p)+2; %2
            else
                Tf(h,p) = T(h,p)-2; 
            end
        else
            if rem(h,2)
                Tf(h,p) = T(h,p)-2; %2
            else
                Tf(h,p) = T(h,p)+2;
            end
        end
        faultystates = [faultystates; P(p),H(h),Tf(h,p)];
    end
end
figure(1)
surf(H,P,T)
% Faulty dataset
figure(2)
surf(H,P,Tf)
% Saving data
save('phT','normalstates','faultystates')