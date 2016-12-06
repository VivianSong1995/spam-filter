%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @author     : Ahmet Alparslan Celik
% @date       : 24.02.2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function test_set_accuracy( training_features, training_labels, ...
                            test_features, test_labels, MT)
%
%   Test-set accuracy as a function of removed number of features
%

    % initialize vectors
    numOfRemovedFeatures = 1:length(MT);        % x axes
    accuracyValues = zeros(1, length(MT));      % y axes
    temp_testLabels = test_labels;
    temp_testFeatures = test_features;
    rowSize = length(temp_testLabels');
    temp_MT = MT;

    % paramater initialization
    alpha = 1;
    T_J0 = sum( training_features(training_labels==0, :) );
    T_J1 = sum( training_features(training_labels==1, :) );
    N_1 = sum( training_labels );
    N = length( training_labels );
    P_spam = N_1 / N;

    % remove one feature at a time and calculate the accuracy
    for ctr = 1:length(MT)
        % determine the least informative element
        [~, index] = min(temp_MT);

        % remove the least informative feature
        T_J0 = [T_J0(1:index-1), T_J0(index+1:end)];
        T_J1 = [T_J1(1:index-1), T_J1(index+1:end)];
        temp_testFeatures = [ temp_testFeatures(:, 1:index-1), temp_testFeatures(:, index+1:end) ];
        temp = [ temp_MT(1:index-1), temp_MT(index+1: end) ];
        temp_MT = temp;

        % Update the parameters using MAP estimation with 
        % a fair Dirichlet prior (alpha)
        Theta_J0 = (T_J0 + alpha) ./ ( sum(T_J0) + alpha * length(T_J0));
        Theta_J1 = (T_J1 + alpha) ./ ( sum(T_J1) + alpha * length(T_J1));

        % posterior probabilities
        MAP_J0 = log( (1-P_spam) * ones( rowSize, 1 ) ) + temp_testFeatures * log(Theta_J0');
        MAP_J1 = log( (P_spam) * ones( rowSize, 1) ) + temp_testFeatures * log(Theta_J1');

        % get the result vector
        resultVector = zeros( rowSize, 1);
        resultVector( MAP_J1 > MAP_J0 ) = 1;

        % calculate accuracies
        vector_temp = resultVector - temp_testLabels;
        accuracy = length(find(vector_temp == 0)) / length(resultVector);

        % store the accuracy
        accuracyValues(ctr) = accuracy;
    end

    % plot the test-set accuracy graph
    figure;                      
    plot(numOfRemovedFeatures, accuracyValues);
    xlabel('# of removed features');   
    ylabel('Accuracy of the model');         
    title('Test-set Accuracy'); 

end

