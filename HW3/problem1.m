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
Ytest = Dataset(:,8);

%change all 0 to -1
Y(~Y) = -1;
Ytest(~Ytest) = -1;

n = size(Dataset,1);    %Wataset size
W(1:n,1) = 1/n;         %initial weights
m = 10;                 %number of iterations
epsilon = 0;
F = 0;
   

for t=0:1
    
    %train the Wecision tree, my weak learner
    tree3 = fitctree(X,Y,'Weights',W,'MinLeafSize',100);

    %weight sum error
    for i=1:n
        prediction = predict(tree3, X(i,:));
        err = signFunc(prediction, Y(i,:));
        epsilon = epsilon + W(i,:)*err;
    end
    fprintf('exiting\n');
    pause(5);
    %alpha term
    alpha = 0.5 * log((1-epsilon)/epsilon);
    F = F + alpha * predict(tree3,Xtest); 
    
    Z =0;
    for i=1:n
        Z = Z + W(i,:)*exp(-alpha*Y(i,:)*predict(tree3, X(i,:)));
    end
    
    for i=1:n
        W(i,:) = (W(i,:)/Z) * exp(-alpha*Y(i,:)*predict(tree3, X(i,:)));
    end
end
F_Sign = sign(F);
view(tree3,'Mode','graph');    

    
    
    
    
    




