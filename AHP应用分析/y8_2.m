clc,clear,close all
A=[1	3	2	5	4	7	8
0.33333	1	2	3	5	6	7
0.5	0.5	1	2	3	5	5
0.2	0.33333	0.5	1	2	3	4
0.25	0.2	0.33333	0.5	1	2	3
0.14286	0.16667	0.2	0.33333	0.5	1	2
0.125	0.14286	0.2	0.25	0.33333	0.5	1];
[v,d]=eig(A);
d(1,1) %特征值
v(:,1) %特征向量



%%
clc,clear,close all
B1=[1	2	3
0.5	1	2
0.33333	0.5	1];
[v,d]=eig(B1) ;
d(1,1) %特征值
v(:,1) %特征向量


%%
clc,clear,close all
B2=[1	5	0.5
0.2	1	0.2
2	5	1];
[v,d]=eig(B2) ;
d(1,1) %特征值
v(:,1) %特征向量

%%
clc,clear,close all
B3=[1	0.125	0.16667
8	1	2
6	0.5	1];
[v,d]=eig(B3) ;
d(1,1) %特征值
v(:,1) %特征向量
%%
clc,clear,close all
B4=[1	3	0.25
0.33333	1	2
4	0.5	1];
[v,d]=eig(B4) ;
d(1,1) %特征值
v(:,1) %特征向量
%%
clc,clear,close all
B5=[1	0.5	0.33333
2	1	0.5
3	2	1];
[v,d]=eig(B5) ;
d(1,1) %特征值
v(:,1) %特征向量

%%
clc,clear,close all
B6=[1	5	3
0.2	1	0.5
0.33333	2	1];
[v,d]=eig(B6) ;
d(1,1) %特征值
v(:,1) %特征向量

%%
clc,clear,close all
B7=[1	4	8
0.25	1	5
0.125	0.2	1];
[v,d]=eig(B7) ;
[v,d]=eig(B7) ;
d(1,1) %特征值
v(:,1) %特征向量


%%
n=7;
Y=[7.2531,3.0092,3.0536,3.0183,4.2312,3.0092,3.0037,3.0940];
CI=(Y-n)/(n-1);
CR=(CI/1.32);
W=[-0.847	0.528	0.093	-0.51	-0.257	-0.928	0.943
-0.466	0.133	0.864	-0.49	-0.466	-0.175	0.32
-0.257	0.839	0.495	-0.707	-0.847	0.329	0.087];
w=[-0.757
-0.495
-0.34
-0.197
-0.134
-0.0814
-0.059];
w3=W*w