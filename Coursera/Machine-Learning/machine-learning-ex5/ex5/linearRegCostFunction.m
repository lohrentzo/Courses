function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%sprintf("X: ")
%disp(size(X))

%sprintf("theta: ")
%disp(size(theta))
%disp(theta)

h = X*theta;


%thetaReg = [0;theta(2:end, :);];
thetaReg = theta;
%disp(thetaReg)
thetaReg(1,:) = 0;
%disp(thetaReg)

J = (1/(2*m))*sum((h-y).^2) + (lambda/(2*m))*(thetaReg'*thetaReg);

%sprintf("X: ")
%disp(X);

grad = (1/m)*(X'*(h-y)) + (1/m)*lambda*thetaReg;

%sprintf("end")





% =========================================================================

grad = grad(:);

end
