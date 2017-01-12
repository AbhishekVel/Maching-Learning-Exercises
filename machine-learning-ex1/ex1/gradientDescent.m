function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

 % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    

for iter = 1:num_iters
    thetaOne = theta(1); thetaTwo = theta(2);
    ho = thetaOne*X(:,1) + thetaTwo*X(:, 2); % 97x1
    thetaOne = thetaOne  - (alpha/m) * (X(:, 1)' * (ho-y));
    thetaTwo = thetaTwo - (alpha/m) * (X(:, 2)' * (ho-y));
    
    theta = [thetaOne; thetaTwo];
    J_history(iter) = computeCost(X, y, theta);
end

end
