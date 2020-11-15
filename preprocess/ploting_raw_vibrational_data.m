%SCRIPT TO PLOT THE RAW DATA FROM THE CWRU
format compact
clear all
clc

load('inner_race_fault_0.021_0_1797.mat');                                 %load all my data files
load('ball_fault_0.021_0_1797.mat');
load('normal_baseline_data_1797.mat');
load('centered_outer_race_fault_0.021_0_1797.mat');


xInner = X209_DE_time;
fsInner = 12000;
tInner = (0:length(xInner)-1)/fsInner;                                     %moving to the right time scale (sample rate 12khz)
figure
plot(tInner, xInner)                                                       %plotting for acceleration vs time
xlabel('Time, (s)')
ylabel('Acceleration (g)')
title('Raw Signal: Inner Race Fault')                                      %zoom in the first 0.1sec to see the form of the singal better
xlim([0 0.1])

[pEnvInner, fEnvInner, xEnvInner, tEnvInner] = envspectrum(xInner, fsInner); %calling matlab comants to move to the frequency domain through en envelope spectrum diagramm
figure
plot(fEnvInner, pEnvInner)                                                 %plotting fro amplitude vs frequency
xlim([0 900])                                                           
ncomb = 10;
helperPlotCombs(ncomb, 162.185)                                            %plotting the fundamental fault frequency of the bearing and its harmonics
xlabel('Frequency (Hz)')
ylabel('Peak Amplitude')
title('Envelope Spectrum: Inner Race Fault')
legend('Envelope Spectrum', 'BPFI Harmonics')

function helperPlotCombs(ncomb, f)                                         %defining the helper function to plot the fault frequencies

ylimit = get(gca, 'YLim');
ylim(ylimit);
ycomb = repmat([ylimit nan], 1, ncomb);
hold(gca, 'on')
for i = 1:length(f)
    xcomb = f(i)*(1:ncomb);
    xcombs = [xcomb; xcomb; nan(1, ncomb)];
    xcombs = xcombs(:)';
    plot(xcombs, ycomb, '--')
end
hold(gca, 'off')
end
