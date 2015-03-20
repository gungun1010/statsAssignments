fid = fopen('dataset2/dataset2.2.train.csv');
out = textscan(fid,'%f%f%s','delimiter',',');
fclose(fid);

feature1 = out{1};
feature2 = out{2};
label = out{3};

X = [feature1, feature2];
Y = zeros(size(label,1),1);
for i=1:size(label,1)
    if strcmp(label(i,1),'True') == 1
        Y(i,1) = 1;
    else
        Y(i,1) = -1;
    end
end
        
figure;
gscatter(X(:,1),X(:,2),Y);
title('Scatter Diagram of Data');

fprintf('start training\n');
gamma = 100;
SVMModel = fitcsvm(X,Y,'Standardize',true,'KernelFunction','RBF',...
    'BoxConstraint',Inf,'ClassNames',[-1,1]);

fprintf('prepare graph\n');
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));

xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(SVMModel,xGrid);

% Plot the data and the decision boundary
fprintf('start plotting\n');
figure;
h(1:2) = gscatter(X(:,1),X(:,2),Y,'rb','.');
hold on
ezpolar(@(x)1);
h(3) = plot(X(SVMModel.IsSupportVector,1),X(SVMModel.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal
hold off
