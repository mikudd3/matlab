clc,clear
load xj2006.mat
% xjm(1,:)=mean(xj2006); xjd(1,:)=std(xj2006);
% xj2006=zscore(xj2006);%数据的标准化
n0=size(xj2006);
j=1;k=1;
for i=1:n0(1,1)
%     if i==1||i==2||i==7||i==8||i==9||i==10||i==11||i==15||i==18||i==19||i==20
    if i==1||i==2||i==7||i==8||i==9||i==10||i==15||i==18
        x2006(j,:)=xj2006(i,:);
        j=j+1;
    end
    if i==3||i==4||i==6||i==12||i==13||i==14||i==16
        y2006(k,:)=xj2006(i,:);
        k=k+1;
    end
end
X=x2006; % X is the input data with N*L
Y=y2006; % Y is the output data with M*L 
n=size(X',1);m=size(X,1);s=size(Y,1);
A=[-X' Y'];b=zeros(n,1);
LB=zeros(m+s,1);UB=[];
for i=1:n;
    Aeq=[X(:,i)' zeros(1,s)];beq=1;
    f=[zeros(1,m) -Y(:,i)'];
    w(:,i)=linprog(f,A,b,Aeq,beq,LB,UB);%输出DMU的最佳权向量
    E(i,i)=Y(:,i)'*w(m+1:m+s,i);%输出DMU的相对效率值Eii
    for k=1:n;
        f=[zeros(1,m) Y(:,k)'];
        Aeq=[X(:,k)' zeros(1,s)
        E(i,i)*X(:,i)' -Y(:,i)'];
        beq=[1;0];
        v=linprog(f,A,b,Aeq,beq,LB,UB);
        E(i,k)=(Y(:,k)'*v(m+1:m+s))/(X(:,k)'*v(1:m));
    end
end
E   %输出最佳权向量
e=mean(E) %输出相对权向量
omega=w(1:m,:)   %投入权向量
mu=w(m+1:m+s,:)  %产出权向量 

n=size(w);
n1=size(x2006);
n2=size(y2006);
vx(1,8)=0;
for i=1:n(1,2)
    for j=1:n1(1,1)
        vx(1,i)=vx(1,i)+w(j,i)*X(j,i);
    end
end
uy(1,8)=0;
for i=1:n(1,2)
    for j=1:n2(1,1)
        uy(1,i)=uy(1,i)+w(j+n1(1,1),i)*Y(j,i);
    end
end
paixu=uy-vx;        
