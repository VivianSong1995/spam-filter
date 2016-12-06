%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @author     : Ahmet Alparslan Celik
% @date       : 24.02.2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function naive_bayes_MLE( training_features, training_labels, ...
                            test_features, test_labels)
%
%   Naive Bayes classifier using MLE 
% 

% Training part
T_J0 = sum( training_features(training_labels==0, :) );
T_J1 = sum( training_features(training_labels==1, :) );
N_1 = sum( training_labels );
N = length( training_labels );
P_spam = N_1 / N;
Theta_J0 = T_J0 ./ sum(T_J0);
Theta_J1 = T_J1 ./ sum(T_J1);

% Classifying using posterior probabilities
% get the # of rows
[rowSize, ~] = size(test_features);

spam_likelihood = log(Theta_J1);
non_spam_likelihood = log(Theta_J0);
resultVector = zeros(rowSize, 1);

testDataSpam = test_features .* repmat(spam_likelihood, rowSize, 1);
testDataNonSpam = test_features .* repmat(non_spam_likelihood, rowSize, 1);
testDataSpam(  isnan(testDataSpam)  ) = 0;
testDataNonSpam(  isnan(testDataNonSpam)  ) = 0;
postA = sum(testDataSpam') + log(P_spam);
postB = sum(testDataNonSpam') + log(1 - P_spam);
resultVector(postA' > postB') = 1;

% percentage of validation
vector1 = resultVector - test_labels;

Q4_3_accuracy = length(find(vector1 == 0)) / length(resultVector)

end

