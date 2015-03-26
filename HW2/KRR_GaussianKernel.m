%this function is the descprtion of gaussian kernel
function K = KRR_GaussianKernel(X,X_test, sigma)
n1sq = sum(X.^2,1);
n1 = size(X,2);
n2sq = sum(X_test.^2,1);
n2 = size(X_test,2);

D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X_test;    
K = exp(-D/(2*sigma^2));

end

