%training data Y for CNN
format compact
clear all
clc

A=ones(19,800);
B=2*ones(19,800);
C=0*ones(19,800);
D=3*ones(19,800);

training_dataY_c=vertcat(A,B,C,D);
save('training_dataY_c.mat','training_dataY_c')