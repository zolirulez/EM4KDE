load SigmaFaulty
load SigmaNormal
load phT
data = normalstates;
load phT_eval
evaldata = [normalstates; faultystates2; faultystates5];
N = length(data);
D = size(data,2);
N_eval = length(evaldata);
if noise
    evaldata(:,3) = evaldata(:,3) + rand(length(evaldata),1)*1-1;
end
SigmaSet = {SigmaNormal,SigmaFaulty};
likelihood = cell(2,1);
delta_data_ver1 = repmat(evaldata,1,N);
for it = 1:N
    delta_data_ver1(:,(it-1)*D+1:it*D) =...
        delta_data_ver1(:,(it-1)*D+1:it*D) - data(it,:);
end
delta_data_ver2 = zeros(size(delta_data_ver1));
for iter2 = 1:D
    delta_data_ver2(:,(iter2-1)*N+1:iter2*N) = delta_data_ver1(:,iter2:D:end);
end
delta_data_ver3 = reshape(delta_data_ver2,N*N_eval,D);
for it = 1:length(SigmaSet)            
    R = chol(SigmaSet{it},'upper');
    gain = 1/((2*pi)^(D/2)*det(R))/N;
    m = delta_data_ver3/R;
    r = reshape(-0.5*sum(m.*m,2),N*N_eval,1);
    lnC = -max(r); % This is not the recommended way
    r_vector = gain*exp(r + lnC);
    likelihood{it} = sum(reshape(r_vector,N_eval,N),2);
end
if noise
    fault_evaluation_noisy = likelihood{2}./(likelihood{1}+likelihood{2});
else
    fault_evaluation = likelihood{2}./(likelihood{1}+likelihood{2});
end