%%%%%%%%%%%%%%%%%%%%%%
%% 2. train readout %%
%%%%%%%%%%%%%%%%%%%%%%

disp('training readout:');

%% main loops
for j = 1:n_learn_loops

    fprintf('  loop: %d/%d, ', j, n_learn_loops);

    % initial conditions
    Xv = 1 * (2 * rand(numUnits, 1) - 1);
    X = tanh(Xv);
    Xv_current = Xv;
    Out = zeros(numOut, 1);

    % Initialize arm state
    x0 = [zeros(2, 1); zeros(2, 1)]; %theta1, theta2, thetadot1, thetadot2
    u = zeros(numOut, 1);
    arm_x = zeros(4, n_steps);
    arm_x(:, 1) = x0 + arm_dynamics(x0, u, L) * dt * arm_dt;

    arm = arm_x(:, 1) + msr_train_noise_amp * randn(4, 1);
    error_cntl = 0;

	train_window = 0;

	for i = 1:n_steps

        Input = input_pattern(:, i);

        %% update RNN units
        Xv_current = W * X + WIn * Input + WFb * Out;
        Xv = Xv + ((-Xv + Xv_current) ./ tau) * dt;
        X = tanh(Xv);

        % output through readout
        Out = WOut * X + alpha * arm(1:2);
        Out_learn_history(:, i, j) = Out;

        %PD control
        error_prev = error_cntl;
        error_cntl = Out - arm(1:2);
        dif = (error_cntl - error_prev) / arm_dt;
        u = Kp * error_cntl + Kd * dif;

        if i < start_train_n
            u = zeros(numOut, 1);
        end

        arm_x(:, i+1) = arm_x(:, i) + arm_dynamics(arm_x(:, i), u, L) * dt * arm_dt;
        arm = arm_x(:, i+1) + msr_train_noise_amp * randn(4, 1);

		% start-end training window
		if i == start_train_n
			train_window = 1;
		end
        if i == end_train_n
			train_window = 0;
		end

		%% train readout
		if (train_window == 1) && (rem(i, learn_every) == 0)

			% update error
            error = Out - target_Out(:, i);

		    P_old = P;
			P_old_X = P_old * X;
			den = 1 + X' * P_old_X;
			P = P_old - (P_old_X * P_old_X') / den;

			% update output weights
            WOut = WOut - error * (P_old_X / den)';             
        end
    end

    % performance
    for n = 1:numOut
        R = corrcoef(Out_learn_history(n, start_train_n:end_train_n, j), target_Out(n, start_train_n:end_train_n));
        R2_learn(n, j) = R(1, 2)^2;
    end

	fprintf('R^2=%.3f\n', R2_learn(1, j));
end
