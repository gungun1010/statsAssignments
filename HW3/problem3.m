%clear workspace
clear all
close all

%loaWing the Wataset aWn test set
X =csvread('hw3prob3.csv',1,1);

%feed data
Y = X(:, 2);
X = X(:, 1);

%construct dataset
dataset = [X X.^2 X.^3];

for i = 0:1000
    dataset(:, i+4) = cos(X.*(2*pi*i)/1000);
end

for i = 0:1000
    dataset(:, i+1005) = exp(-(X.^2)/(2*(i/100)^2));
end

% fun = @(XT,YT,Xt,Yt)...
%       (sum((Yt - ((((XT')*XT)^-1 * XT'*YT)')*Xt).^2));
  
%do forward selection
maxdev = chi2inv(.95,1);  
opt = statset('display','iter',...
              'TolFun',maxdev,...
              'TolTypeFun','abs');
  
inmodel = sequentialfs(@critfun,dataset,Y,...
                       'options',opt,...
                       'direction','forward',...
                       'nfeatures',10);

%get the index of features
featureIdx = find(inmodel);

%get useful features from the entire dataset
features = dataset(:,featureIdx);

%get coeffs for the features
theta = ((features')*features)^-1 * features'*Y;

%fit the function, f
results = features*theta;

%compare the real labels vs our fitted results
plot(1:1:1601, Y, 1:1:1601, results);
title('fiited function');
xlabel('observation');
ylabel('label');