clear all
close all

%load data
fid = fopen('dataset2/dataset2.4.csv');
out = textscan(fid,'%f%f','delimiter',',');
fclose(fid);

feature = out{1};
label = out{2};

fid = fopen('dataset2/dataset2.4.test.csv');
out = textscan(fid,'%f%f','delimiter',',');
fclose(fid);

feature_test = out{1};
label_test = out{2};

%using gaussian kernel
n = 100;
sigma = sqrt(0.1);
lamda = 1/n;
X = feature(1:n,1);
Y = label(1:n,1);
K = KRR_GaussianKernel(X',X',sigma);

%Regulation matrix
Identity=eye(length(K(:,1)));

%using closed form of f_hat
KRR_train = ((K+lamda*Identity)^-1)*Y;

X_test = feature(1:n,1);
Y_test = label(1:n,1);
KRR_test = KRR_GaussianKernel(X', X_test', sigma);

%prediction here is the f_hat we are trying to fit
prediction = KRR_test'*KRR_train;
MSE = (1/n)*norm(prediction-Y_test).^2;
figure 
plot(prediction,'bx','LineWidth',2)
hold on

plot(Y_test,'ro','LineWidth', 0.7)
%hold off
legend('prediction','actual label')
title('Validation Results')

figure
plot(MSE, 'LineWidth',2);