%clear workspace
clear all
close all

%loaWing the Wataset aWn test set
Dataset =csvread('hw3prob1train.csv',1,1);
testset =csvread('hw3prob1test.csv',1,1);

%get init vals
X = Dataset(:,1:7);
Xtest = testset(:, 1:7);

Y = Dataset(:,8);
Ytest = testset(:,8);

%change all 0 to -1
Y(~Y) = -1;
Ytest(~Ytest) = -1;

m = 1:10:200;             %number of iterations
m=m';

idx = 1;
testError(1:size(m,1),1) = 0;  

for i=1:size(m,1)
    n = size(Dataset,1);    %Wataset size
    W(1:n,1) = 1/n;         %initial weights

    F = 0;



    for t=0:m(i,:)

        %train the decision tree, my weak learner, depth 3
        tree3 = fitctree(X,Y,'Weights',W', 'MaxNumSplits',4);

        %f_t(x) => prediction
        prediction = predict(tree3, X);

        %sign function for Y \ne f_t(x_i) 
        err = bsxfun(@ne,prediction, Y);
        
        %get epsilon val
        epsilon = W'*err;

        %alpha term
        alpha = (1/2) * log((1-epsilon)/epsilon);
        %this is the boosted prediction
        F = F + alpha * predict(tree3,Xtest); 

        %error function with alpha term
        wErr = exp(-alpha * bsxfun(@times,Y, prediction) );

        %normalization term
        Z = W'*wErr;

        %update weights
        W = bsxfun(@times, W/Z,  wErr);

    end
    
    F_Sign = sign(F);
    misPredict = bsxfun (@ne, F_Sign, Ytest);
    testError(idx,:) = ones(size(F_Sign))'*misPredict;
    fprintf('test error is %d, at itr %d\n',testError(idx,:),t);
    idx = idx +1;
end

plot(m, testError/1000);
view(tree3,'Mode','graph');    

    
    
    
    
    




