# Dependencies

Here are project dependencies that are installed automatically during `vagrant up`:

+ python==3.5.2
+ numpy==1.13.3
+ pandas==0.21.0
+ scikit-learn==0.19.1
+ matplotlib==2.1.0
+ seaborn==0.8.1
+ jupyter

# Model Selection

The two python notebook files in ./Notebooks/ analyze and build upon well known ideas in data analysis and machine learning to build a classification machine that has both "good" predictive performance and an efficient runtime. Here are some important observations gained from this analysis:

1. The data contains 14 feature columns and 1 class label column.
2. Fortunately, there are no missing values that we have to interpolate or ignore.
3. The class distribution in the training set is heavily unbalanced. There are roughly 3 times as many samples for "Mitt Romney" than for "Barak Obama".	
	+ This may or may not be a sampling fluke; in either case, we have to take precautions when building our model to ensure that the higher probability of one class does not factor into the bias of the entire model (i.e. instead of predicting on features, it predicts on label distributions).
4. Some features such as 'total population' and 'total households' are positively correlated. Others such as 'household size' and 'median age' are negatively correlated.
5. 'Total Households' and 'Total Population' do not have much variance and may not provide much information to our model.

A more appropriate metric for analyzing the performance of our model may be the 'f1 score'. It uses the ratio between precision and recall to compute a value between 0 and 1 (higher is better). This will produce a more meaningful result that factors in the bias. The confusion matrix is another representation of this
idea and will help us see true positives and true negatives (along the diagonal) and false positives and false negatives (along the anti-diagonal). From this we can reasonably predict how the model will perform under unknown conditions such as inference.

For both (4) and (5), applying Principle Component Analysis (PCA) will capture the greatest amount of variance and help mitigate the effect of correlated variables.
See further work for additional feature engineering ideas.

For PCA we can see that roughly 8 features capture between 95-98% of the variance in the data. While PCA has great advantages such as reducing computational load and producing better results, it minimizes the interpretability of the model and with that increasing debugging time along with difficulty improving the model.

We scale the features to zero mean and unit variance to speed up the convergence of the model. Furthermore, the labels are encoded into binary values.

Below are some of the models that were tested and the rational for choosing or rejecting them:

### Naive Bayes Classifier
PROS:
+ Easy to build and interpret results
+ Works well with few samples
+ Training is very quick. Hyperparameter optimization is possible using log-marginal likelyhood
+ Uses Bayesian statistics (i.e. Gives the probabilities of each class label as an expectation of the result)

CONS:
+ Assumes independence (We fix this by performing PCA and inputting features in orthogonal basis).

### Random Forrest Classifier
PROS:
+ Is an ensemble method which uses 'divide and conquer' by applying several weaker algorithms (Decision Trees). This allows it to be fast and quite scalable to larger datasets
+ Can handle non-linear relationships well
+ Most of the time just works with neural-net level performance
+ Can handle missing data well (not relavent for this project)
+ During experimentation achieved the best "accuracy" and training time without overfitting

CONS:
+ Difficult to optimize. There are many hyperparameters to adjust. The optimization section of the notebook uses a grid search to find the best combination of parameters which can take a long time
+ Has the potential to overfit in some situations

### Gaussian Process Classifier
PROS:
+ Relies on Bayesian statistics (i.e. Gives the probabilities of each class label as an expectation of the result)
+ Hyperparameter optimization is possible using log-marginal likelihood
+ No assumption about independence
+ Elagent way to think about data as events with discrete probabilities and deviations from the mean

CONS:
+ Data is assumed to follow gaussian distribution (which in most cases is true)
+ Computationally expensive to train. Especially with Radial Basis Function (RBF) kernel. No way to efficiently handle sparse input.

While the f1 scores of the three models were comparable, the Random Forrest produced the least false signals and was chosen to further optimize.

The optimization procedure was done as an exercise to maximize cross validatation metrics on the RF model by tuning hyperparameters. While this was somewhat successfull (~1% improvement in f1), the rate of overfitting increased drastically. The optimal parameters leverage the bias in the dataset and produce more false signals. Build_model.py uses the default Random Forest parameters in the sklearn library.

# Further Work

Given time constraints and computing resources, there are a few improvements that can be made to this project.

1. Better documentation and project structure
2. Feature engineering to remove variables with low variance and correlated features
3. Experiement with other models and techniques such as Support Vector Machines (SVM), ensemble methods, and regularization (to prevent overfitting)
4. Optimize RF using a custom scoring algorithm using precision, recall, and area under ROC curve on a bigger/smarter parameter set
5. Further research
