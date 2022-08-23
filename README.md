# Titanic-Classification

## Introduction

This is a classification project using Python and the Titanic data base. Our goal was to use different machine learning algorithms to predict which passengers survived and compare the results to determine which algorithm is best suited for this problem. The algorithms we tested were naive Bayes, decision tree, random forest, k nearest neighbors, logistic regression, support vector machines and neural network. The ones with the best performance were decision tree, random forest and logistic regression and the results for these three algorithms were not statistically distinct from each other.

One of the most well known classification problems is that of the Titanic. The ship famously sunk during it's first voyage due to a collision with an iceberg, leading to the deaths of most of it crew and passengers. The idea behind the projects is, therefore, to use machine learning to predict which passengers would survive, based on the available passenger data. We repeated this process using different classification algorithms to find which one is the best fit for this particular problem.

The data set we will be using is the *train.csv* file (which we renamed *titanic.csv*) obtained from https://www.kaggle.com/competitions/titanic/data?select=train.csv at 15:02 of 30/06/2022.

## Preliminary Analysis

We can load the CSV file into Python using the Pandas library. Before we use the machine learning algorithms, let us take a look at our data.

The file accounts for 891 of the passengers. Out of these, only 342 (approximately 38%) survived. Creating a tree map relating the attributes "Survived", "Sex" and "Pclass", we can see that sex was very significant in determining who survived, with most of the survivors (approximately 68%) being women. We can also see that most of the dead (approximately 68%) were in the third class, a class which accounted for approximately 55% of the total number of passengers.

Looking at each of the columns, we conclude that the "PassengerId" and "Name" attributes are unlikely to help the algorithms determine which of the passengers survived, since the values of these two columns never repeat themselves. Therefore,we will not be using these two attributes.

Finally, we can see that there are some missing values in the data, specifically in the "Age", "Cabin" and "Embarked" columns. We will deal with these in the preprocessing step.
