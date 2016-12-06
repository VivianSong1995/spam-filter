%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @author     : Ahmet Alparslan Celik
% @date       : 24.02.2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function spam_mail_ratio( training_labels )
%
%   Calculate the spam data ratio over the training dataset
%

% Since it is binary data, summation can be used
Q4_2_spam_percentage = sum(training_labels) / length(training_labels)

end

