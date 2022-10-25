function [Sw,Sb,mu_i,ci,W]=LDA(X,X_label,class_number)
% this function is used to calculate Sw and Sb
%{
 Input:
      X            - 训练样本
      class_number - 样本种类，有多少类
      X_label      - 样本的具体类别
 Output:
      Sw - the within-class matrix
      Sb - the between-class matrix
      mu_i 样本均值
      ci - 每个类别的类别编号
      W  -最优判别矩阵
%}

X_label_uni = unique(X_label);

if isa(X_label_uni, 'cell')%判断参数是否为指定类型对象，如果是返回1，如果不是返回0
    X_label_uni = string(X_label_uni);
end
feature_number = size(X,2);%训练样本的列
 

Sw = zeros(feature_number,feature_number);
%Sb = zeros(feature_number,feature_number);
mu_i = zeros(class_number,size(X,2));
ci = zeros(class_number,1);
for i=1:class_number
    mu_i(i,:)=mean(X(X_label==X_label_uni(i),:));%先判断X_l与X_L_uni是否相等，返回0与1，并取均值
    ci(i)=sum(X_label==X_label_uni(i));%求和
    SS=(sum(X_label==X_label_uni(i))-1)*cov(X(X_label==X_label_uni(i),:));
    Sw=Sw+SS; % covariance matrix is 1/li times by Sw which obtained by loop strategy
    %{ 
    using the definition to calculate Sb
    bb=sum(X_label==X_label_uni(i)).*((mu_i(i,:)-mean(X))'*(mu_i(i,:)-mean(X)));
    Sb=Sb+bb;
    %}
end

Sb=(class_number-1)*cov(mu_i); % the same as the result obtained by its definition

[W1,D]=eig(Sw\Sb);
[~,index_w] = sort(diag(D),'descend');
W = W1(:,index_w);

end