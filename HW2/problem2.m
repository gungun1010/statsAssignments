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

SVMModel1 = fitcsvm(X,Y,'KernelFunction','gaussianKernenl','Standardize',true);
