%testing data Y for CNN
format compact
clear all
clc

A=ones(151,800);
B=2*ones(151,800);
C=0*ones(151,800);
D=3*ones(151,800);
E=vertcat(A,B,C,D);
E(605,:)=3;

save('testing_dataY_c.mat','E')

