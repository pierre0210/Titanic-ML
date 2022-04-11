# Titanic prediction

## 原始結果
* score: 0.76555
* change test size to 1/4

## 改善方式
* Method 1 調整test size & random state (result: 0.76555)
* Method 2 改變訓練資料集
* Method 3 改變訓練方式

## test log
* ~~age category changed~~ split categories using qcut function
* added embark categories
* ~~name category~~
* combine categories
* combine two dataframe

https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
https://www.kaggle.com/competitions/titanic/discussion/311970
https://yulongtsai.medium.com/https-medium-com-yulongtsai-titanic-top3-8e64741cc11f