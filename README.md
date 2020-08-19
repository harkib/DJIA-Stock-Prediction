# DJIA-Stock-Perdiction

Data-set contains the top 25 headline of the day (per day from 2008 to 2016) and labels (binary) representing whether the Dow Jones Industrial Average went up or down. This repo feeds the headline as block of text through the appropriate bert pre-processing steps and uses uncased base BERT to predict the movement of the Dow Jones Industrial Average. We also consider sentiment analysis (vaderSentiment and textblob) combined with previous day stock values and several statisitics based models.

|BERT: Results|
|-|
|Epoch:0, loss: 0.6959441122255827, Accuracy: 0.5079365079365079|
|Epoch:1, loss: 0.6973660707473754, Accuracy: 0.5079365079365079|
|Epoch:2, loss: 0.6930920927148116, Accuracy: 0.5079365079365079|
|Epoch:3, loss: 0.7107855194493344, Accuracy: 0.5079365079365079|


```
-------------- LinearDiscriminantAnalysis --------------
              precision    recall  f1-score   support

           0       0.35      0.50      0.41       130
           1       0.66      0.51      0.57       248

    accuracy                           0.51       378
   macro avg       0.50      0.50      0.49       378
weighted avg       0.55      0.51      0.52       378
```
```
------------------ SVM Classification ------------------
              precision    recall  f1-score   support

           0       0.11      0.48      0.18        44
           1       0.88      0.50      0.64       334

    accuracy                           0.50       378
   macro avg       0.50      0.49      0.41       378
weighted avg       0.79      0.50      0.59       378

```
```
------------------- SGDClassifier ---------------------
              precision    recall  f1-score   support

           0       0.30      0.58      0.40        98
           1       0.79      0.54      0.64       280

    accuracy                           0.55       378
   macro avg       0.55      0.56      0.52       378
weighted avg       0.66      0.55      0.58       378
```
```
----------------- KNeighborsClassifier ----------------
              precision    recall  f1-score   support

           0       0.55      0.51      0.53       201
           1       0.48      0.52      0.50       177

    accuracy                           0.51       378
   macro avg       0.51      0.51      0.51       378
weighted avg       0.52      0.51      0.51       378
```
```
-------------- GaussianProcessClassifier ---------------
              precision    recall  f1-score   support

           0       0.41      0.52      0.46       149
           1       0.62      0.52      0.57       229

    accuracy                           0.52       378
   macro avg       0.52      0.52      0.51       378
weighted avg       0.54      0.52      0.52       378
```
```
--------------- RandomForestClassifier ----------------
              precision    recall  f1-score   support

           0       0.28      0.47      0.35       113
           1       0.69      0.49      0.57       265

    accuracy                           0.49       378
   macro avg       0.48      0.48      0.46       378
weighted avg       0.57      0.49      0.51       378
```
