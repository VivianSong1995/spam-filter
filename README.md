# spam-filter
Naive bayes classifier

**main.m**   
There are five function but the function "main" should be called. The other functions are automatically called.
```matlab	
		spam_mail_ratio(training_labels)
		naive_bayes_MLE( training_features, training_labels, ...
   	                         	test_features, test_labels);
   	 	naive_bayes_MAP( training_features, training_labels, ...
                            	test_features, test_labels);
    	mutual_information( training_features, training_labels );
    	test_set_accuracy( training_features, training_labels, ...
                            	test_features, test_labels, MT);
                              
```
**function main( path_to_dataset )**  
There is only one parameter. The parameter should justify the following format "/home/xyz/datasets" and indicates the location of the dataset files.  
It return nothing. But it prints the results to the console with meaningful labels.

Depending on the question number, the meaning of the outputs are the followings

	"spam_mail_ratio" func. returns "Q4_2_spam_percentage"
		For the question 4.2., it prints the percentage
		
	"naive_bayes_MLE" func. returns "Q4_3_accuracy"
		For the question 4.3., it prints the accuracy after using
		Naive Bayer classifier using MLE
		
	"naive_bayes_MAP" func. returns "Q4_4_accuracy"
		For the question 4.4., it prints the accuracy after using
		Naive Bayer classifier using MAP with a prior alpha=1
		
	"mutual_information" func. returns "Q4_5_maxTenValues" &  "Q4_5_maxTenIndexes"
		For the question 4.5., it prints the top ten informative elements 
		and their indexes. However, since there exists many NaN value in 
		the result set, according to how NaN values are handled there are 
		two different results. These differences are provided and explained
		in the report of this homework.
		
		In this function, during the calculation of the mutual information, 
		in order not to get NaN values, whole matrix is incremented by 1 so 
		that all feature have at least the value 1; therefore, there exists no
		NaN values.
	
	"test_set_accuracy" func. returns a plot
		For the question 4.6., this part provides a plot of Test-set accuracy 
		as a function of removed number of features. In the mutual information
		generation part, in order to handle NaN values, according to how these NaN
		values are handled, the resulting graph may show some minor differences. 
		These differences are provided and explained in the report of this homework.
		
		In this function, during the calculation of the mutual information, 
		in order not to get NaN values, whole matrix is incremented by 1 so 
		that all feature have at least the value 1; therefore, there exists no
		NaN values.
