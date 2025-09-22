%%%%%%%%%%%%%
%% 3. test %%
%%%%%%%%%%%%%

disp('testing:');

time_window = 0;

for j = 1:n_test_loops

    fprintf('  loop: %d/%d,\n', j, n_test_loops);

    % initial conditions
    Xv = 1 * (2 * rand(numUnits, 1) - 1);
    X = tanh(Xv);
    Xv_current = Xv;
    Out = zeros(numOut, 1);

    % Initialize arm state
    x0 = [zeros(2, 1); zeros(2, 1)]; %theta1, theta2, thetadot1, thetadot2
    u = zeros(numOut, 1);
    u_in = u;

    arm_x = zeros(4, n_steps);
    arm_x(:, 1) = x0 + arm_dynamics(x0, u_in, L) * dt * arm_dt;

    arm = arm_x(:, 1) + msr_test_noise_amp * randn(4, 1);
    error_cntl = 0;

    for i = 1:n_steps

        if rem(i, 100) == 0
            fprintf('    step: %d/%d\n', i, n_steps);
        end

        Input = input_pattern(:, i);

        % update RNN units
        Xv_current = W * X + WIn * Input + WFb * Out;
        Xv = Xv + ((-Xv + Xv_current) ./ tau) * dt;
        X = tanh(Xv);

        % output through readout
        Out = WOut * X + alpha * arm(1:2);
        Out_test_history(:, i, j) = Out;
        
        %PD control
        error_prev = error_cntl;
        error_cntl = Out - arm(1:2);
        dif = (error_cntl - error_prev) / arm_dt;
        u = Kp * error_cntl + Kd * dif;

        if i < start_train_n
            u = zeros(numOut, 1);
        end

        error_history(:, i, j) = error_cntl;
        u_history(:, i, j) = u;

        if strcmp(perturbation, 'impulse')
            if i == (start_train_n + 500)
                u = u - 0.5 * [1.0; 1.0];
            end
        elseif strcmp(perturbation, 'obstacle')
            if arm_x(1, i) >= 0.5 && i <= (start_train_n + 500)
                u = zeros(numOut, 1);
                arm_x(3:4, i) = zeros(numOut, 1);
            end
        end

        % System noise
        % if i >= start_train_n
        %     u = u + sys_noise_amp * randn(2, 1);
        % end

        arm_x(:, i+1) = arm_x(:, i) + arm_dynamics(arm_x(:, i), u, L) * dt * arm_dt;
        arm = arm_x(:, i+1) + msr_test_noise_amp * randn(4, 1);

        Out_test_history_link2(:, i, j) = culc_kinematics(arm_x(1:2, i), L);
        Out_test_history_link1(:, i, j) = culc_kinematics_elbow(arm_x(1:2, i), L);
    end

    % performance
    for n = 1:numOut
        R = corrcoef(Out_test_history(n, start_train_n:end_train_n, j), target_Out(n, start_train_n:end_train_n));
        R2_test(n, j) = R(1, 2)^2;
    end
    fprintf('R^2=%.3f\n', R2_test(1, j));
end

R_ave = mean(R2_test, 2);
R_std = std(R2_test');
fprintf('  mean R^2=%.3f\n', R_ave);

max_u2 = max(u_history(2, 500:1000, 1));
fprintf('  peak torque = %.5f\n', max_u2);

