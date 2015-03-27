%clear workspace
clear all
close all

%open the training dataset
fid = fopen('dataset2/dataset2.2.train.csv');
out = textscan(fid,'%f%f%s','delimiter',',');
fclose(fid);

feature1 = out{1};
feature2 = out{2};
label = out{3};

X = [feature1, feature2];
Y = zeros(size(label,1),1);

%open the testing dataset
fid = fopen('dataset2/dataset2.2.test.csv');
out = textscan(fid,'%f%f%s','delimiter',',');
fclose(fid);

feature1_test = out{1};
feature2_test = out{2};
label_test = out{3};

X_test = [feature1_test, feature2_test];
Y_test = zeros(size(label_test,1),1);

for i=1:size(label,1)
    if strcmp(label(i,1),'True') == 1
        Y(i,1) = 1;
    else
        Y(i,1) = -1;
    end
end

for i=1:size(label_test,1)
    if strcmp(label_test(i,1),'True') == 1
        Y_test(i,1) = 1;
    else
        Y_test(i,1) = -1;
    end
end

figure;
gscatter(X_test(:,1),X_test(:,2),Y_test);
title('Scatter Diagram of Data');

fprintf('start training\n');

sigma = 0.01;
gamma = 1/(2*sigma^2);

% Libsvm options
% -s 0 : classification
% -t 2 : RBF kernel
% -g : gamma in the RBF kernel
model = svmtrain(Y, X, sprintf('-s 0 -t 2 -g %g', gamma));

[predicted_label, accuracy, decision_values] = svmpredict(Y, X, model);

% Plot heatmap
plotboundary(Y, X, model, 't');

title(sprintf('\\sigma = %g', sigma), 'FontSize', 14);
