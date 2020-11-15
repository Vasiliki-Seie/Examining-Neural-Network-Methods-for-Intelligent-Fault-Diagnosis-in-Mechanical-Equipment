%preproccess for cnn
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

training_data_i=transpose(training_data_i(:,1:8:end));                     %decimate every 8th column from these matrices
training_data_b=transpose(training_data_b(:,1:8:end));
training_data_n=transpose(training_data_n(:,1:8:end));
training_data_c_o=transpose(training_data_c_o(:,1:8:end));


training_dataX=vertcat(training_data_n,training_data_i,training_data_b,training_data_c_o);
a=size(training_dataX,1);
save('training_dataX.mat','training_dataX')

for i=1:a
    b=a/6;
    c=2*a/6;
    d=3*a/6;
    training_dataY(1:b,1)=double(1);
    training_dataY(b+1:c,1)=double(2);
    training_dataY(c+1:d,1)=double(0);
    training_dataY(d+1:a,1)=double(3);
end

save('training_dataY.mat','training_dataY')

    
    
    
    
    
    
    


