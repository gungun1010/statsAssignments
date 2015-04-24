%clear workspace
clear all
close all

%loaWing the Wataset aWn test set
Dataset =csvread('hw3prob2train.csv',1,1);
testset =csvread('hw3prob2test.csv',1,1);

X = Dataset(:,1:6);
Xtest = testset(:, 1:6);

Y = Dataset(:,7);
Ytest = Dataset(:,7);
Yreal = Dataset(:,8);
YrealTest = testset(:,8);

%change all 0 to -1
Y(~Y) = -1;
Ytest(~Ytest) = -1;

Ftrain = 0;
F = 0;
m = 1:1:200;                 %number of iterations
m=m';
idx = 1;

residual = Yreal;

testError(1:size(m,1),1) = 0;  
trainError(1:size(m,1),1) =0;

for i=1:size(m,1)
    for t=0:m(i,:)
        %fit a weak learner
        tree3 = fitrtree(X,residual);

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
    
    residualTest_sign = sign(residualTest);
    misPredict = bsxfun (@ne, residualTest_sign, Ytest);
    testError(idx,:) = ones(size(residualTest_sign))'*misPredict;

    residual_sign = sign(residual);
    misPredictTrain = bsxfun (@ne, residual_sign, Y);
    trainError(idx,:) = ones(size(residual_sign))'*misPredictTrain;
    
    idx = idx +1;
    display(m(i,1));
end


% 
% predTrain = sign(predTrain);
% misPredict = bsxfun (@ne, predTrain, Y);
% 
% error = ones(size(predTrain))'*misPredict;
