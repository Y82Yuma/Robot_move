clear all;

%% main parameters
task = 'point'; % point or butterfly
perturbation = 'impulse'; % impulse or obstacle or none

interval = 1000;
g = 1.5; % spectral radius
numUnits = 1000; % number of units
noise_amp = 0;
msr_train_noise_amp = 0; % measurement noise in training
msr_test_noise_amp = 0; % measurement noise in testing
sys_noise_amp = 0; % system noise 0.02;

alpha = 1;
Iter = 1;
PLOT = 1;

% parameter setting
param_RC_robot;

for iter = 1:Iter
    fprintf('\n====== %d/%d ======\n', iter, Iter);

    % construct RC
    construct_network;

    % train readout
    train_RC_robot;

    % test
    test_RC_robot;

    % plot
    if PLOT == 1
        plot_RC_robot;
        video_RC_robot;
    end
end
