%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @author     : Ahmet Alparslan Celik
% @date       : 24.02.2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function naive_bayes_MAP( training_features, training_labels, ...
                            test_features, test_labels)
%
%   Extended Naive Bayes classifier using MAP estimate of 
%   theta parameters using a fair Dirichlet prior (alpha)
%

% get the # of rows
[rowSize, ~] = size(test_features);

alpha = 1;
% Training part
T_J0 = sum( training_features(training_labels==0, :) );
T_J1 = sum( training_features(training_labels==1, :) );
N_1 = sum( training_labels );
N = length( training_labels );
P_spam = N_1 / N;
Theta_J0 = (T_J0 + alpha) ./ ( sum(T_J0) + alpha * length(test_features));
Theta_J1 = (T_J1 + alpha) ./ ( sum(T_J1) + alpha * length(test_features));

% Posterior probabilities
MAP_J0 = log( (1-P_spam) * ones(rowSize,1) ) + test_features * log(Theta_J0');
MAP_J1 = log( (P_spam) * ones(rowSize,1) ) + test_features * log(Theta_J1');

resultVector = zeros(rowSize, 1);
resultVector( MAP_J1 > MAP_J0 ) = 1;

% percentage of validation
vector1 = resultVector - test_labels;
Q4_4_accuracy = length(find(vector1 == 0)) / length(resultVector)

end

