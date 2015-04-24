%clear workspace
clear all
close all

%loaWing the Wataset aWn test set
Dataset =csvread('hw3prob2train.csv',1,1);
testset =csvread('hw3prob2test.csv',1,1);

X = Dataset(:,1:6);
Xtest = testset(:, 1:6);

Y = Dataset(:,7);
Ytest = testset(:,7);

Yreal = Dataset(:,8);
YrealTest = testset(:,8);

%change all 0 to -1
Y(~Y) = -1;
Ytest(~Ytest) = -1;

Ftrain = 0;
F = 0;
m = 1:1:40;                 %number of iterations
m=m';
idx = 1;

testError(1:size(m,1),1) = 0;  
trainError(1:size(m,1),1) =0;

for i=1:size(m,1)
    
    
    residual = Yreal;

    F =0;
    Ftrain = 0;
    for t=0:m(i,:)
        %fit a weak learner
        tree3 = fitrtree(X,residual, 'MaxNumSplits',4);

        %get training result from the weak learner 
        predTrain = predict(tree3, X);

        %use the same tree for testing
        predTest = predict(tree3, Xtest);

        %sum all past iteration up to eventually converge
        Ftrain = Ftrain + predTrain;
        F = F + predTest;

        %compute the residual
        residual = bsxfun(@minus,Yreal,Ftrain);
        residualTest = bsxfun(@minus, YrealTest, F);
    end
%        testError(idx,:) = sum(residualTest);
%        trainError(idx,:) = sum(residual);
     
    residualTest_sign = sign(residualTest);
    misPredict = bsxfun (@ne, residualTest_sign, Ytest);
    testError(idx,:) = sum(misPredict);

    residual_sign = sign(residual);
    misPredictTrain = bsxfun (@ne, residual_sign, Y);
    trainError(idx,:) = sum(misPredictTrain);
    
    idx = idx +1;
    display(m(i,1));
end

figure;
plot(m, testError);
title('testing error');
xlabel('iteration');
ylabel('error rate');

figure;
plot(m, trainError);
title('training error');
xlabel('iteration');
ylabel('error rate');
% 
% predTrain = sign(predTrain);
% misPredict = bsxfun (@ne, predTrain, Y);
% 
% error = ones(size(predTrain))'*misPredict;
