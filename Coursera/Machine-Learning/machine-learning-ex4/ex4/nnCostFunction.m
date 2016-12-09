function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1
%disp(size(X))
a1 = [ones(m, 1) X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(rows(a2), 1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

%disp(size(a3))

%Part 2
%sprintf("y: ")
%disp(y)
I = eye(num_labels);
%disp(I)
Y = zeros(m, num_labels);
%disp(size(Y))
for i=1:m
%sprintf("y(i): ");
%disp(y(i));
%disp(I(y(i), :))
  Y(i, :)= I(y(i), :);
end
%sprintf("Y: ")
%disp(Y);

%J = (1/m)*sum(sum())
Jsum = 0;
for i=1:m
  for k=1:num_labels
    Jsum += -Y(i, k)*log(a3(i,k)) - (1-Y(i,k))*log(1-a3(i,k));
  end
end

J = (1/m)*Jsum;

regFactor1 = 0;
regFactor2 = 0;

%disp(size(Theta1))
%disp(size(Theta2))

rs = rows(Theta1);
cs = columns(Theta1);

for j=1:rs
  for k = 2:cs
    regFactor1 += Theta1(j,k)^2;
  end
end

rs = rows(Theta2);
cs = columns(Theta2);

for j=1:rs
  for k = 2:cs
    regFactor2 += Theta2(j,k)^2;
  end
end

regFactorSum = regFactor1+regFactor2;

J = J + regFactorSum*lambda/(2*m);

% Part 3
Delta_1 = 0;
Delta_2 = 0;
%disp(size(X))
%disp(size(y))

for t=1:m
  % 1.
  a1 = X(t, :)';
  a1 = [1; a1];
  z2 = Theta1*a1;
  a2 = sigmoid(z2);
  a2 = [1; a2];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  %disp(size(a3))
  %disp(size(Y))
  % 2.
  d3 = zeros(size(a3));
  for k=1:num_labels
    d3(k) = a3(k) - Y(t, k);
  end

  %disp(size(Theta2))
  %disp(size(Theta2'*d3))
  z2 = [1;z2];
  %disp(size( z2 ))

  d2 = Theta2'*d3.*sigmoidGradient(z2);
  d2 = d2(2:end);

  Delta_1 = Delta_1 + d2*a1';
  Delta_2 = Delta_2 + d3*a2';

end
Theta1_grad = Delta_1./m;
Theta2_grad = Delta_2./m;

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad += (lambda/m).*Theta1;
Theta2_grad += (lambda/m).*Theta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
