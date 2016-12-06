%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @author     : Ahmet Alparslan Celik
% @date       : 24.02.2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ MT ] = mutual_information( training_features, ...
                                            training_labels )
%
%   Mutual Information and rank features
%
[docSize, wordSize] = size(training_features);

temp_training_feature = training_features;
temp_training_feature( temp_training_feature>0 ) = 1;

% in order to handle NaN values, bias alpha=1 is added to
% the summation matrixes
alpha = 1;
N_11 = sum( temp_training_feature(training_labels==1, :) ) + alpha;
N_10 = sum( temp_training_feature(training_labels==0, :) ) + alpha;
N_01 = (sum(training_labels)*ones(1,wordSize)) - ...
        sum( (temp_training_feature(training_labels==1, :) )) + alpha;
N_00 = ((docSize - sum(training_labels))*ones(1,wordSize)) - ...
        sum( (temp_training_feature(training_labels==0, :) )) + alpha;
N = N_11 + N_10 + N_01 + N_00;
N_0 = N_01 + N_00;
N_1 = N_10 + N_11;

MT = (N_11./N).*log2(N.*N_11 ./ (N_1.^2)) + ...
    (N_01./N).*log2(N.*N_01 ./ (N_0.*N_1)) + ...
    (N_10./N).*log2(N.*N_10 ./ (N_1.*N_0)) + ...
    ((N_00./N).*log2(N.*N_00 ./ (N_0.*N_0)));

[~, indexes] = sort(MT,'descend'); 

maxTenMI = indexes(1:10);

% prints the top ten informative elements and their indexes
Q4_5_maxTenMIValues = MT(maxTenMI)
Q4_5_maxTenMIIndexes = maxTenMI
MT
end

