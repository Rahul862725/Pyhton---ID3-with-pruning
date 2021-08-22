# Pyhton---ID3-with-pruning


In Machine learning, ID3 (Iterative Dichotomiser 3) is an algorithm invented by Ross Quinlan. used to generate a decision tree from a dataset. 
Main purpose of this project to add pruning and k-fold cross validation algo so that the efficency of this algorithm increase and it will give best
result for future dataset.
The ID3 algorithm begins with the original set S as the root node. On each iteration of the algorithm, it iterates through every unused attribute of the set S and calculates the entropy or the information gain  of that attribute. It then selects the attribute which has the smallest entropy (or largest information gain) value. The set S is then split or partitioned by the selected attribute to produce subsets of the data.The algorithm continues to recurse on each subset, considering only attributes never selected before. In sort

Calculate the entropy of every attribute a of the data set  S.
Partition ("split") the set S into subsets using the attribute for which the resulting entropy after splitting is minimized or, equivalently, information gain is maximum.
Make a decision tree node containing that attribute.
Recurse on subsets using the remaining attributes.

In last tree will made based on that dataset. Based on this tree, Future dataset will predicted.

This is algorithm of Id3 algo. But if dataset is very large then this lead to overfitting and also time complexity of testing increase. To overcome from overfiting , a concept pruning is used. Pruning is a technique in machine learning and search algorithms that reduces the size of decision trees by removing sections of the tree that provide little power to classify instances. Pruning reduces the complexity of the final classifier, and hence improves predictive accuracy by the reduction of overfitting. 

But this is one tree in whole sample space of tree, i can not say it is more efficent (in turms of testing future dataset) tree in whole possible trees . To find the best tree among from sample space, i use a machine learning algorithm k-fold cross validation algorithm. 
In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The cross-validation process is then repeated k times, with each of the k subsamples used exactly once as the validation data. The k results can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling (see below) is that all observations are used for both training and validation, and each observation is used for validation exactly once.

so as a result by applying k-fold and purning algorithm, Id3 algorithm become more efficent and free from overfiting.
