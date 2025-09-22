function output = arm_dynamics(x, u, L)
    m1 = 0.5;
    m2 = 0.5;
    b1 = 0.5;
    b2 = 0.5;

    l1 = L(1);
    l2 = L(2);

    % input arguments
    torque = [u(1); u(2)];
    theta1 = x(1);
    theta2 = x(2);
    theta1dot = x(3);
    theta2dot = x(4);

    % inertia matrix
    M11 = (m1 + m2) * l1^2 + m2 * l2^2 + 2 * m2 * l1 * l2 * cos(theta2);
    M12 = m2 * l2^2 + m2 * l1 * l2 * cos(theta2);
    M21 = m2 * l2^2 + m2 * l1 * l2 * cos(theta2);
    M22 = m2 * l2^2;
    M = [M11 M12; M21 M22];

    % Coriolis force vector
    V1 = - m2 * l1 * l2 * (2 * theta1dot * theta2dot + theta2dot^2) * sin(theta2);
    V2 = m2 * l1 * l2 * theta1dot^2 * sin(theta2);
    V = [V1; V2];

    % frition (viscosity) vector
    F = [b1 * theta1dot; b2 * theta2dot];

    thetaddot = inv(M) * (torque - F - V);
    output = [theta1dot; theta2dot; thetaddot(1); thetaddot(2)];
end