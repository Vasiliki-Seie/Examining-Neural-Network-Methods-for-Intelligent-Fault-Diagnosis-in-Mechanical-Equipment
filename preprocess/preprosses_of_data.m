%PREPROSSESING OF THE VIBRATIONAL DATA (TRAINING DATASETS)
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
    training_data_i(:,i)=X209_DE_time(l:k);
    training_data_b(:,i)=X222_DE_time(l:k);
    training_data_n(:,i)=X097_DE_time(l:k);
    training_data_c_o(:,i)=X234_DE_time(l:k);
end

training_data_i=training_data_i(:,1:8:end);                                %decimate every 8th column from these matrices
training_data_b=training_data_b(:,1:8:end);
training_data_n=training_data_n(:,1:8:end);
training_data_c_o=training_data_c_o(:,1:8:end);

training_data_i=reshape(training_data_i,15200,1);                          %reshape them in 15200*1 vectors
training_data_b=reshape(training_data_b,15200,1);
training_data_n=reshape(training_data_n,15200,1);
training_data_c_o=reshape(training_data_c_o,15200,1);

training_dataX=vertcat(training_data_i,training_data_b,training_data_n,training_data_c_o);  %and finally vertically merge them together to have all the train data in a 60800*1 vector
training_dataX=rescale(training_dataX,-1,1);                               %linearly normalize the X data 

a=size(training_dataX);  
a=a(1);

for i=1:a                                                                  %constract a Y data vector with the labeles corresponding to each data point
    b=a/4;
    c=a/2;
    d=a*3/4;
    training_dataY(1:b,1)=double(1);
    training_dataY(b+1:c,1)=double(2);
    training_dataY(c+1:d,1)=double(0);
    training_dataY(d+1:a,1)=double(3);
end

fileIDx=fopen('training_dataX.txt','w');                                   %print the data in txt files to use in pyhton enviroment
fprintf(fileIDx,'%.4f\r\n',training_dataX);
fileIDy=fopen('training_dataY.txt','w');
fprintf(fileIDy,'%d\r\n',training_dataY);
