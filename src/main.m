%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @author     : Ahmet Alparslan Celik
% @date       : 24.02.2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function main( path_to_dataset )

% Load the datasets
test_features = load( strcat( path_to_dataset, '/test-features.txt' ));
test_labels = load( strcat( path_to_dataset, '/test-labels.txt' ));
training_features = load( strcat( path_to_dataset, '/train-features.txt' ));
training_labels = load( strcat( path_to_dataset, '/train-labels.txt' ));

%----------------------
%
%   Calculate the spam data ratio over the training dataset
%

spam_mail_ratio(training_labels);

%----------------------
%
%   Naive Bayes classifier using MLE 
% 
%

naive_bayes_MLE( training_features, training_labels, ...
                            test_features, test_labels);
                        
%----------------------
%
%   Extended Naive Bayes classifier using MAP estimate of 
%   theta parameters using a fair Dirichlet prior (alpha)
%

naive_bayes_MAP( training_features, training_labels, ...
                            test_features, test_labels);

%----------------------
%
%   Mutual Information and rank features
%

MT = mutual_information( training_features, training_labels );

%----------------------
%
%   Test-set accuracy as a function of removed number of features
%

test_set_accuracy( training_features, training_labels, ...
                            test_features, test_labels, MT);

                        
end
% End of the CS464-1 HW1 main function

