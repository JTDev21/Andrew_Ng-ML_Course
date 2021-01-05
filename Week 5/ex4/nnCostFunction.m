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
m = size(X, 1); % 5000 x 400
         
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

% =================================================
% Theta1 - 25 x 401
% Theta2 - 10 x 26

% Theta1_grad - 25 x 401
% Theta2_grad - 10 x 26

% X - 5000 x 400
% y - 5000 x 1
% =================================================

% for i = 1:m
%     a1 = [1 X(i, :)]'; % n x 1 ~ 401 x 1 || Insert Bias Unit
    
%     z2 = Theta1 * a1;
%     a2 = sigmoid(z2); % 25 x 1
%     a2 = [1 ; a2]; % 26 x 1 || Insert Bias Unit
    
%     z3 = Theta2 * a2;
%     a3 = sigmoid(z3); % 10 x 1

%     y_vec = zeros(num_labels, 1); % 10 x 1
%     y_vec(y(i)) = 1;

%     J = J + sum(-y_vec' * log(a3) - (1 .- y_vec)' * log(1 .- a3));

% % Part 2: Implement the backpropagation algorithm to compute the gradients
% %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
% %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
% %         Theta2_grad, respectively. After implementing Part 2, you can check
% %         that your implementation is correct by running checkNNGradients
% %
% %         Note: The vector y passed into the function is a vector of labels
% %               containing values from 1..K. You need to map this vector into a 
% %               binary vector of 1's and 0's to be used with the neural network
% %               cost function.
% %
% %         Hint: We recommend implementing backpropagation using a for-loop
% %               over the training examples if you are implementing it for the 
% %               first time.
% %
%     y_vec = zeros(num_labels, 1); % 10 x 1
%     y_vec(y(i)) = 1;

%     delta_3 = a3 - y_vec;
%     delta_2 = (Theta2(:, 2:end)' * delta_3) .* sigmoidGradient(z2);

%     Theta2_grad = Theta2_grad + (delta_3 * a2');
%     Theta1_grad = Theta1_grad + (delta_2 * a1');
% endfor;

% % Part 3: Implement regularization with the cost function and gradients.
% %
% %         Hint: You can implement this around the code for
% %               backpropagation. That is, you can compute the gradients for
% %               the regularization separately and then add them to Theta1_grad
% %               and Theta2_grad from Part 2.
% %

% % Regularization
% regularization = (lambda/(2*m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
% J = (J/m) + regularization;

% % Backpropagation (w/ Regularization)
% theta1_reg = Theta1;
% theta2_reg = Theta2;

% theta1_reg(:, 1) = 0;
% theta2_reg(:, 1) = 0;

% Theta1_grad = ((1/m) * Theta1_grad) + ((lambda/m * theta1_reg));
% Theta2_grad = ((1/m) * Theta2_grad) + ((lambda/m * theta2_reg));








% =================================================
% Theta1 - 25 x 401
% Theta2 - 10 x 26

% Theta1_grad - 25 x 401
% Theta2_grad - 10 x 26

% X - 5000 x 400
% y - 5000 x 1
%=================================================================================
%========Vectorized Implementation for unregularized and unregularized cost functions================================

% Cost function (regularized)
Y = zeros(num_labels, m); % 10 x m
Y(sub2ind(size(Y), y', 1:m)) = 1;

a1 = [ones(1, m) ; X']; % 401 x m

z2 = Theta1 * a1; % 25 x m
a2 = [ones(1, m) ; sigmoid(z2)]; % 26 x m

z3 = Theta2 * a2; % 10 x m
a3 = sigmoid(z3);

% J-cost
J = ((1/m) * sum(sum(-Y .* log(a3) - (1 .- Y) .* log(1 .- a3))));

% Adding regularized error, ignore bias units
J = J + (lambda/(2*m)) * sum(sum(Theta1(:, 2:end) .^ 2, 2));
J = J + (lambda/(2*m)) * sum(sum(Theta2(:, 2:end) .^ 2, 2));

% ====== Backpropagation ===========
delta_3 = a3 .- Y;
delta_2 = (Theta2(:, 2:end)' * delta_3) .* sigmoidGradient(z2);

Theta1_grad = delta_2 * a1';
Theta2_grad = delta_3 * a2';

theta1_reg = Theta1; theta1_reg(:, 1) = 0;
theta2_reg = Theta2; theta2_reg(:, 1) = 0;

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * theta1_reg;
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * theta2_reg;

%=================================================================================
%=================================================================================





% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end;