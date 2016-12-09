function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
j_0 = zeros(m,1);
j_1 = zeros(m,1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    theta_0 = theta(1);
    theta_1 = theta(2);

    %fprintf("Iteration: %d\n", iter);
    %fprintf("theta_0: %f\n", theta_0);
    %fprintf("theta_1: %f\n", theta_1);

    for i = 1:m
      j_0(i) = theta_0 + theta_1*X(i,2) - y(i);
      j_1(i) = j_0(i)*X(i,2);
    end

    theta_0 =  theta_0 - alpha * 1/m * sum(j_0);
    theta_1 =  theta_1 - alpha * 1/m * sum(j_1);

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

    %disp(J_history(iter));

    theta(1) = theta_0;
    theta(2) = theta_1;
end

end
