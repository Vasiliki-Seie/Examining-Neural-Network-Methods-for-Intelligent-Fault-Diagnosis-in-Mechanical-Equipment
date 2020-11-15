%preprocecess of testing for cnn
format compact
clear all
clc

load('inner_race_fault_0.021_0_1797.mat');                                 %load all my data files
load('ball_fault_0.021_0_1797.mat');
load('normal_baseline_data_1797.mat');
load('centered_outer_race_fault_0.021_0_1797.mat');

for i=1:605                                                                %organise every data file in a 605*200 matrix
    k=200*i; 
    l=k-199;
    testing_data_i(:,i)=X209_DE_time(l:k);
    testing_data_b(:,i)=X222_DE_time(l:k);
    testing_data_n(:,i)=X097_DE_time(l:k);
    testing_data_c_o(:,i)=X234_DE_time(l:k);
end

testing_dataX=vertcat(transpose(testing_data_n),transpose(testing_data_i),transpose(testing_data_b),transpose(testing_data_c_o));
a=size(testing_dataX,1);
save('testing_dataX.mat','testing_dataX');

for i=1:a
    b=a/6;
    c=2*a/6;
    d=3*a/6;
    testing_dataY(1:b,1)=double(1);
    testing_dataY(b+1:c,1)=double(2);
    testing_dataY(c+1:d,1)=double(0);
    testing_dataY(d+1:a,1)=double(3);
end

save('testing_dataY.mat','testing_dataY');