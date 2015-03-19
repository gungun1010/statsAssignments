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

fK01= zeros(1,steps);
nk1 = 0.1;
ThetaGD = zeros(size(dataset,2)-1,1); % k=0
for k=1:steps
     ThetaGD = ThetaGD - (nk1 / m) * x.'*(x*ThetaGD-y);
     fK01(:,k) = norm(ThetaGD - ThetaHat)^2;
end

fK05= zeros(1,steps);
nk5 = 0.5;
ThetaGD = zeros(size(dataset,2)-1,1); % k=0
for k=1:steps
     ThetaGD = ThetaGD - (nk5 / m) * x.'*(x*ThetaGD-y);
     fK05(:,k) = norm(ThetaGD - ThetaHat)^2;
end

fK10= zeros(1,steps);
nk10 = 1;
ThetaGD = zeros(size(dataset,2)-1,1); % k=0
for k=1:steps
     ThetaGD = ThetaGD - (nk10 / m) * x.'*(x*ThetaGD-y);
     fK10(:,k) = norm(ThetaGD - ThetaHat)^2;
end

fK100= zeros(1,steps);
nk100 = 10;
ThetaGD = zeros(size(dataset,2)-1,1); % k=0
for k=1:steps
     ThetaGD = ThetaGD - (nk100 / m) * x.'*(x*ThetaGD-y);
     fK100(:,k) = norm(ThetaGD - ThetaHat)^2;
end

figure;
plot(fK001);
xlabel('steps');
ylabel('margin');
title('step size = 0.01');

figure;
plot(fK01);
xlabel('steps');
ylabel('margin');
title('step size = 0.1');

figure;
plot(fK05);
xlabel('steps');
ylabel('margin');
title('step size = 0.5');

figure;
plot(fK10);
xlabel('steps');
ylabel('margin');
title('step size = 1');

figure;
plot(fK100);
xlabel('steps');
ylabel('margin');
title('step size = 10');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  part d) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
v1=std(x(:,1));
v2=std(x(:,2));
vy=std(y(:,1));
xn = [x(:,1) ./ v1, x(:,2) ./ v2];
yn = y./vy;

%create grid for theta1 and theta2
theta1=linspace(-40,40);
theta2=linspace(-40,40);

[Theta1,Theta2]=meshgrid(theta1,theta2);
f_Theta = zeros(size(Theta1));

% plug into LMS function
for i=1:m
    f_Theta = f_Theta+( xn(i,1).*Theta1 + xn(i,2).*Theta2 - yn(i,1) ).^2 ;
end
f_Theta=(1/(2*m)).*f_Theta;

% organize the plot
contour(Theta1, Theta2, f_Theta);
grid on
axis equal
axis tight
xlabel('theta1');
ylabel('theta2');
title('problem3, part d), level curve of f_Theta');

%do the optimization as above
ThetaHat = ((xn.'*xn)^(-1))*xn.'*yn;
display(ThetaHat);

steps = 60;
fK001 = zeros(1,steps);
nk01 = 0.01;
ThetaGD = zeros(size(dataset,2)-1,1); % k=0
for k=1:steps
     ThetaGD = ThetaGD - (nk01 / m) * xn.'*(xn*ThetaGD-yn);
     fK001(:,k) = norm(ThetaGD - ThetaHat)^2;
end

figure;
plot(fK001);
xlabel('steps');
ylabel('margin');
title('step size = 0.01, normalized');
