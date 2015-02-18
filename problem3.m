%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  part a) %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%loading the dataset
dataset =csvread('dataset1.csv',0,0);
m = size(dataset,1);

%get  (x, y) pairs
x = dataset(:,1:2);
y = dataset(:,3);

%create grid for theta1 and theta2
theta1=linspace(-40,40);
theta2=linspace(-40,40);

[Theta1,Theta2]=meshgrid(theta1,theta2);
f_Theta = zeros(size(Theta1));

% plug into LMS function
for i=1:m
    f_Theta = f_Theta+( x(i,1).*Theta1 + x(i,2).*Theta2 - y(i,1) ).^2 ;
end
f_Theta=(1/(2*m)).*f_Theta;

% organize the plot
contour(Theta1, Theta2, f_Theta);
grid on
axis equal
axis tight
xlabel('theta1');
ylabel('theta2');
title('problem3, part a), level curve of f_Theta');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  part b) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
ThetaHat = ((x.'*x)^(-1))*x.'*y;
display(ThetaHat);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  part c) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
steps = 60;
fK001 = zeros(1,steps);
nk01 = 0.01;
ThetaGD = zeros(size(dataset,2)-1,1); % k=0
for k=1:steps
     ThetaGD = ThetaGD - (nk01 / m) * x.'*(x*ThetaGD-y);
     fK001(:,k) = norm(ThetaGD - ThetaHat)^2;
end

% steps = 20;
% fK01= zeros(1,steps);
% nk1 = 0.1;
% ThetaGD = zeros(size(dataset,2)-1,1); % k=0
% for k=1:steps
%      ThetaGD = ThetaGD - (nk1 / m) * x.'*(x*ThetaGD-y);
%      fK01(:,k) = norm(ThetaGD - ThetaHat)^2;
% end
% 
% steps = 10;
% fK05= zeros(1,steps);
% nk5 = 0.5;
% ThetaGD = zeros(size(dataset,2)-1,1); % k=0
% for k=1:steps
%      ThetaGD = ThetaGD - (nk5 / m) * x.'*(x*ThetaGD-y);
%      fK05(:,k) = norm(ThetaGD - ThetaHat)^2;
% end
% 
% steps = 10;
% fK10= zeros(1,steps);
% nk10 = 1;
% ThetaGD = zeros(size(dataset,2)-1,1); % k=0
% for k=1:steps
%      ThetaGD = ThetaGD - (nk10 / m) * x.'*(x*ThetaGD-y);
%      fK10(:,k) = norm(ThetaGD - ThetaHat)^2;
% end
% 
% steps = 10;
% fK100= zeros(1,steps);
% nk100 = 10;
% ThetaGD = zeros(size(dataset,2)-1,1); % k=0
% for k=1:steps
%      ThetaGD = ThetaGD - (nk100 / m) * x.'*(x*ThetaGD-y);
%      fK100(:,k) = norm(ThetaGD - ThetaHat)^2;
% end

figure;
plot(fK001);

% figure;
% plot(fK01);
% 
% figure;
% plot(fK05);
% 
% figure;
% plot(fK10);
% 
% figure;
% plot(fK100);
% 
% display(fK05);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  part d) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
v1=var(x(:,1));
v2=var(x(:,2));

xn = [x(:,1) ./ v1,x(:,2) ./ v2];

%create grid for theta1 and theta2
theta1=linspace(-40,40);
theta2=linspace(-40,40);

[Theta1,Theta2]=meshgrid(theta1,theta2);
f_Theta = zeros(size(Theta1));

