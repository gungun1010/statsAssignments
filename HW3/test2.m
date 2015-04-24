%clear workspace
clear all
close all

n = 100;
m = 10;
X = rand(n,m);
b = [1 0 0 2 .5 0 0 0.1 0 1];
Xb = X*b';
p = 1./(1+exp(-Xb));
N = 50;
y = binornd(N,p);

Y = [y N*ones(size(y))];

temp = ((X')*X)^-1 * X'*y;

temp2 = X*temp;
sum((y - X*temp).^2)
% maxdev = chi2inv(.95,1);     
% opt = statset('display','iter',...
%               'TolFun',maxdev,...
%               'TolTypeFun','abs');
% 
% inmodel = sequentialfs(@critfun,X,Y,...
%                        'cv','none',...
%                        'nullmodel',true,...
%                        'options',opt,...
%                        'direction','forward');