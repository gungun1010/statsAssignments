%clear workspace
clear all
close all

%open the training dataset
fid = fopen('dataset2/dataset2.1.train.csv');
out = textscan(fid,'%f%f%f','delimiter',',');
fclose(fid);

feature1 = out{1};
feature2 = out{2};
label = out{3};

X = [feature1, feature2];
Y = zeros(size(label,1),1);

%open the test dataset
fid = fopen('dataset2/dataset2.1.test.csv');
out = textscan(fid,'%f%f%f','delimiter',',');
fclose(fid);

feature1_test = out{1};
feature2_test = out{2};
label_test = out{3};

X_test = [feature1_test, feature2_test];

% mark positive label as 1 and negative label as -1
for i=1:size(label,1)
    if label(i,1) >= 0
        Y(i,1) = 1;
    else
        Y(i,1) = -1;
    end
end

%plot the raw data to visualize dataset
figure;
gscatter(X(:,1),X(:,2),Y);
title('Scatter Diagram of Data');

%using gaussian kernel for training
%sigma is used in Gaussian Kernel
%lamda is the regularization parameter
sigma = 3;
K = KRR_GaussianKernel(X',X',sigma);

%Regulation matrix
Identity=eye(length(K(:,1)));

%using closed form of alpha derived from previous question
%label is the Y
%KRR_train is my alpha_hat
lamda = 0.01;
KRR_train = ((K+lamda*Identity)^-1)*label;

%test the KRR
KRR_test = KRR_GaussianKernel(X', X_test', sigma);

% make a prediction, A.K.A: fit the f_hat
prediction = KRR_test'*KRR_train;

%plot the fitted f_hat (prediction) vs real test labels
figure 
plot(prediction,'bx','LineWidth',2)
hold on

plot(label_test,'ro','LineWidth', 2)
%hold off
legend('prediction','actual label')
title(sprintf('\\sigma = %g, \\lambda = %g', sigma, lamda), 'FontSize', 14);
