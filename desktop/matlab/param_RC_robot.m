%% parameters and task definition

%%%%% recurrent neural networks %%%%%%
p_connect = 0.1;				% sparsity parameter (probability of connection)
scale = g / sqrt(p_connect * numUnits);	% scaling for the recurrent matrix
numIn = 1;	                % number of input units
numOut = 2;						% number of output units for the task

% input parameter
input_weight_amp = 1.0;

feedback_weight_amp = 3.0;

% firing rate model
tau = 10.0;						% time constant (ms)
                
% training & loops
learn_every = 2;      % skip time points
n_learn_loops = 10;   % number of training loops
n_test_loops = 1;           % number of test loops

% recursive least squares
delta = 1;                   % P matrix initialization

% robot control
L = [1.8 1.8];
arm_dt = 1;
Kp = 0.04; %0.04;
Kd = 0.25; %0.15 (butterfly); 0.25 (point); 0.45 (point, all);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% input parameters
input_pulse_value = 1.0;
start_pulse = 200;		% (ms)
reset_duration = 50;	% (ms)

% training duration
start_train = start_pulse + reset_duration;
%interval = 1000;
end_train = start_train + interval;

% output parameters
ready_level = 0.2;
peak_level = 1;
peak_width = 30;

% numerics
dt = 1;					% numerical integration time step
tmax = end_train + 200;
n_steps = fix(tmax / dt);			% number of integration time points
time_axis = [0:dt:tmax-dt];

% input function
start_pulse_n = round(start_pulse / dt);
reset_duration_n = round(reset_duration / dt);
start_train_n = round(start_train / dt);

input_pattern = zeros(numIn, n_steps);
input_pattern(1, start_pulse_n:(start_pulse_n + reset_duration_n - 1)) = input_pulse_value * ones(1, reset_duration_n);


% target output function
end_train_n = round(end_train / dt);

if strcmp(task, 'point')
    target_Out = ones(numOut, n_steps);
elseif strcmp(task, 'butterfly')
    numButterfly = interval/1000;
    butt = @(theta) 9 - sin(theta) + 2*sin(3*theta) + 2*sin(5*theta) - sin(7*theta) + 3*cos(2*theta) - 2*cos(4*theta);
    numT = interval / numButterfly / dt;
    thetas = linspace(0, 2 * pi, numT);
    xout = zeros(1, numT);
    yout = zeros(1, numT);
    for i = 1:length(thetas)
        xout(i) = butt(thetas(i)) * cos(thetas(i)) / 14.4734; % 14.4734 is max r
        yout(i) = butt(thetas(i)) * sin(thetas(i)) / 14.4734;
    end
    target_Out_pos = ones(numOut, n_steps);
    target_Out_pos(1, :) = target_Out_pos(1, :) .* xout(1);
    target_Out_pos(2, :) = target_Out_pos(2, :) .* yout(1);

    %repeat
    target_Out_pos(1, (start_train+1):end_train) = repmat(xout, 1, numButterfly);
    target_Out_pos(2, (start_train+1):end_train) = repmat(yout, 1, numButterfly);
    target_Out = culc_inv_kinematics(target_Out_pos, L);
end