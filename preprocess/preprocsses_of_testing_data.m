%PREPROSSESING OF THE VIBRATIONAL DATA (TESTING DATASETS)
format compact
clear all
clc

load('inner_race_fault_0.021_0_1797.mat');                                 %load all my data files
load('ball_fault_0.021_0_1797.mat');
load('normal_baseline_data_1797.mat');
load('centered_outer_race_fault_0.021_0_1797.mat');

testing_data_i=X209_DE_time(1:121000,1);                                   %organise every data file in a 121000*1 vector
testing_data_b=X222_DE_time(1:121000,1);
testing_data_n=X097_DE_time(1:121000,1);
testing_data_o_c=X234_DE_time(1:121000,1);

testing_dataX=vertcat(testing_data_i,testing_data_b,testing_data_n,testing_data_o_c);  %and finally vertically merge them together to have all the testing data in a 484000*1 vector
testing_dataX=rescale(testing_dataX,-1,1);                                             %linearly normalize the X data

a=size(testing_dataX);
a=a(1);

for i=1:a                                                                  %constract a Y data vector with the labeles corresponding to each data point
    b=a/4;
    c=a/2;
    d=a*3/4;
    testing_dataY(1:b,1)=double(1);
    testing_dataY(b+1:c,1)=double(2);
    testing_dataY(c+1:d,1)=double(0);
    testing_dataY(d+1:a,1)=double(3);
end


fileIDx=fopen('testing_dataX.txt','w');                                    %print the data in txt files to use in pyhton enviroment
fprintf(fileIDx,'%.4f\r\n',testing_dataX);
fileIDy=fopen('testing_dataY.txt','w');
fprintf(fileIDy,'%d\r\n',testing_dataY);

