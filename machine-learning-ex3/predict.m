function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

for i = 1 : m
  a1 = [1; X(i,:)'];      % 401 x 1
  z2 = Theta1 * a1;       % (25 x 401)(401 x 1)
  a2 = [1; sigmoid(z2)];  % 26 x 1
  z3 = Theta2 * a2;       % (10 x 26)(26 x 1)
  h = sigmoid(z3);        % a(3)
  [v, cls] = max(h);
  p(i) = cls;
end

% =========================================================================

end