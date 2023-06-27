  # Project Repository - Machine Learning for Text Classification

This repository includes a series of Jupyter notebooks that perform a step-by-step machine learning project. The first number in each notebook represents the sequence of the overall process, and the subsequent numbers represent the variations of that process.

## Notebooks

0. [Quick EDA](./Quick%20EDA.ipynb) - A quick exploratory data analysis of our dataset.

1. [Mark Duplicative Complaints](./Mark%20Duplicative%20Complaints.ipynb) - Process to identify and mark duplicative complaints in the dataset.

  - 1.5 [Text Normalization And Train Test Dev Split](./Text%20Normalization%20And%20Train%20Test%20Dev%20Split.ipynb) - Text normalization process and splitting the data into train, test, and development sets.

2. 2.0 - 2.3 These notebooks cover the iterative updates and evaluations of the TFIDF Vectorizer, including different n-gram approaches and feature selection strategies.

3. The tuning notebooks describe the process of hyperparameter tuning for various machine learning models, including Decision Trees, Logistic Regression, Naive Bayes, Random Forest, and SVM (both linear and non-linear).

  - 3.1.1 - 3.2.2.1.1 These notebooks present strategies for handling imbalanced classes in the dataset, including techniques like SMOTE and Upsampling, while using different classifiers. even it says "best results so far" do not trust it, cause they were all tuned again in the tunning notebooks where the best parameters were investigated. This category is more of trial and error.

  - 3.TEST These notebooks include the final testing of various models.

4. [10-Fold CV For Tuned Models](./10-Fold%20CV%20For%20Tuned%20Models.ipynb) - Evaluation of final models using a 10-Fold Cross Validation. The best model turned out to be the Multinomial Naive Bayes classifier.

5. [NAIVE BAYES Feature Importance Analysis](./NAIVE%20BAYES%20Feature%20Importance%20Analysis.ipynb) - A deeper analysis of the most important features for the Naive Bayes classifier.

6. [NAIVE BAYES Sensitivity and Failure Analysis](./NAIVE%20BAYES%20Sensitivity%20and%20Failure%20Analysis.ipynb) - A sensitivity and failure analysis of the Naive Bayes classifier.

7. [Unsupervised Learning](./unsupervised%20learning.ipynb) - This notebook, presents the unsupervised learning process and its associated analysis.

## Datasets

The datasets used in this project are large, so we've provided a sample of 100 rows from each of the main datasets used in these notebooks:

- [Sample 100 CFPB Data With Duplicate Marked.csv](./Sample%20100%20CFPB%20Data%20With%20Duplicate%20Marked.csv)

- [Sample 100 Raw CFPB Data.csv](./Sample%20100%20Raw%20CFPB%20Data.csv)

- [Sample 100 cfpb_dev.csv](./Sample%20100%20cfpb_dev.csv)

- [Sample 100 cfpb_train.csv](./Sample%20100%20cfpb_train.csv)

Please note, you might need to change the path according to your setup.

## Other Files

- [requirements.txt](./requirements.txt) - The required libraries to run the notebooks in this repository.

- [tfidf_vectorizer_train_split_33k.pkl](./tfidf_vectorizer_train_split_33k.pkl) - The finalized TFIDF Vectorizer used for both supervised and unsupervised tasks.

## Contribution
