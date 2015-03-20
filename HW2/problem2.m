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

sigma = 1;
gamma = 1/(2*sigma^2);
model = svmtrain(Y, X, sprintf('-s 0 -t 2 -g %g', gamma));

[predicted_label, accuracy, decision_values] = svmpredict(Y, X, model);

% Plot heatmap
plotboundary(Y, X, model, 't');

title(sprintf('\\sigma = %g', sigma), 'FontSize', 14);
