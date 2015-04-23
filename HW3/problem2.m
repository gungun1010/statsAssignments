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

F = 0;
residual = Yreal;
for i = 0:100
    
    
    tree3 = fitrtree(X,residual);
    prediction = predict(tree3, X);
    F = F + prediction;
    
    %compute the residual
    residual = bsxfun(@minus,Yreal,F);
    display(norm(residual))
end

% 
% prediction = sign(prediction);
% misPredict = bsxfun (@ne, prediction, Y);
% 
% error = ones(size(prediction))'*misPredict;
