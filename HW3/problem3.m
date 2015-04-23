%clear workspace
clear all
close all

%loaWing the Wataset aWn test set
Dataset =csvread('hw3prob3.csv',1,1);

X = Dataset(:,1);
Y = Dataset(:, 2);
X2 = X.^2;
X3 = X.^3;
