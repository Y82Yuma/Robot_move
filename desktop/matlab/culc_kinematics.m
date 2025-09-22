function xy = culc_kinematics(theta, L)

theta1 = theta(1, :);
theta2 = theta(2, :);

l1 = L(1);
l2 = L(2);

xy = [0 + l1 * cos(theta1) + l2 * cos(theta1 + theta2);...
    -2 + l1 * sin(theta1) + l2 * sin(theta1 + theta2)];

end