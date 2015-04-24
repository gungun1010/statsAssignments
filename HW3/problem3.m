%clear workspace
clear all
close all

%loaWing the Wataset aWn test set
X =csvread('hw3prob3.csv',1,1);

Y = X(:, 2);
X = X(:, 1);

dataset = [X X.^2 X.^3];

for i = 0:1000
    dataset(:, i+4) = cos(X.*(2*pi*i)/1000);
end

for i = 0:1000
    dataset(:, i+1005) = exp(-(X.^2)/(2*(i/100)^2));
end

% [Xtrain, idx] = datasample(dataset, 1200, 'Replace',false);
% 
% idxTest = setdiff(1:1:1601, idx);
% 
% Ytrain = Y(idx,:);
% 
% Xtest = dataset(idxTest,:);
% Ytest = Y(idxTest,:);

fun = @(XT,YT,Xt,Yt)...
      (sum((Yt - ((((XT')*XT)^-1 * XT'*YT)')*Xt).^2));
  
maxdev = chi2inv(.95,1);  
opt = statset('display','iter',...
              'TolFun',maxdev,...
              'TolTypeFun','abs');
  
inmodel = sequentialfs(@critfun,dataset,Y,...
                       'options',opt,...
                       'direction','forward',...
                       'nfeatures',10);
                   
featureIdx = find(inmodel);

features = dataset(:,featureIdx);

theta = ((features')*features)^-1 * features'*Y;

err = sum((Y - features*theta).^2);

