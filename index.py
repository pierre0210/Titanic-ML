import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

df_train = pd.read_csv("data/train_data_titanic.csv")
df_test = pd.read_csv("data/test.csv")
df_data = df_train.append(df_test)

df_data.head()
#df.info()
df_data.describe().T

df_data.isnull().sum() # missing data count

df_data.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
df_data.head()

#sns.pairplot(df[['Survived', 'Fare']], dropna=True)
df_data.isnull().sum()>(len(df_train)/2) # missing value more than average

df_data['Age'].isnull().value_counts() # find missing data
df_data.groupby('Sex')['Age'].median() # find median

df_data['Age'] = pd.qcut(df_data['Age'], 4)
label = LabelEncoder()
df_data['Age'] = label.fit_transform(df_data['Age'])

df_data['Fare'].value_counts()
df_data['Fare'].fillna(df_data['Fare'].median(), inplace=True)

df_data['Fare'] = pd.qcut(df_data['Fare'], 4)
label = LabelEncoder()
df_data['Fare'] = label.fit_transform(df_data['Fare'])

df_data['Embarked'].fillna(df_data['Embarked'].value_counts().idxmax(), inplace=True) # fill with max option & modify df['Embarked'] itself
df_data['Embarked'] = df_data['Embarked'].map({ "S": 0, "C": 1, "Q": 2 })
#df.isnull().sum() # check again

df_data['Family'] = df_data['SibSp'] + df_data['Parch']
df_data.drop(labels=['SibSp', 'Parch'], axis=1, inplace=True)

df_data['Title'] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
#print(df_data['Title'].value_counts())
df_data['Title'] = df_data['Title'].replace(['Capt', 'Col', 'Countess', 'Don',
                                               'Dr', 'Dona', 'Jonkheer', 
                                                'Major','Rev','Sir'],'Rare') 
df_data['Title'] = df_data['Title'].replace(['Mlle', 'Ms', 'Mme'],'Miss')
df_data['Title'] = df_data['Title'].replace(['Lady'],'Mrs')
df_data['Title'] = df_data['Title'].map({"Mr": 0, "Rare" : 1, "Master" : 2,"Miss" : 3, "Mrs" : 4 })
##print(df_data['Title'].value_counts())
df_data.drop(labels=['Name'], axis=1, inplace=True)
Ti = df_data.groupby('Title')['Age'].median()

df_data = pd.get_dummies(data=df_data, columns=['Sex'])
#df_data.drop(['Sex_female'], axis=1, inplace=True)
df_data.drop(['Sex_male'], axis=1, inplace=True)
df_data.head()
#print(df_data.info)

#print(df_data.isnull().sum())

df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]

#print(df_test.isnull().sum())

#print(df.corr())
X = df_train.drop(['Survived', 'Pclass', 'PassengerId', "Embarked"], axis=1)
#X = df.drop(['Survived', 'Pclass', 'PassengerId', 'Parch'], axis=1)
Y = df_train['Survived']

xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.10, random_state=100) #logistic regression 0.25 67

# Logistic Regression
#lr = LogisticRegression(max_iter=3000)
#lr.fit(xTrain, yTrain)

# Decision Tree
#dt = DecisionTreeClassifier(random_state=10)
#dt.fit(xTrain, yTrain)

#Random Forest
rf = RandomForestClassifier(n_estimators=250, random_state=2, min_samples_split=10, oob_score=True)
rf.fit(X, Y)

#result
prediction = rf.predict(xTest)
#prediction = dt.predict(xTest)
#prediction = knn.predict(xTest)
#print('accuracy_score: ', accuracy_score(prediction, yTest))
#print('recall_score: ', recall_score(prediction, yTest))
#print('precision_score: ', precision_score(prediction, yTest))
print('oob_score: ', rf.oob_score_)
pd.DataFrame(confusion_matrix(yTest, prediction), columns=['Predict not Survived', 'Predict Survived'], index=['True not Survived', 'True Survived'])

importances = pd.DataFrame({'feature':xTrain.columns,'importance':np.round(rf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar()

forSubmissionDF = pd.DataFrame(columns=['PassengerId', 'Survived'])
forSubmissionDF['PassengerId'] = df_test['PassengerId']
xSubmit = df_test.drop(labels=['PassengerId', 'Survived', 'Pclass', "Embarked"], axis=1)
submitPrediction = rf.predict(xSubmit).astype(int)
forSubmissionDF['Survived'] = submitPrediction
print(forSubmissionDF)
forSubmissionDF.to_csv('for-submission-20220410-6.csv', index=False)

#export training module
'''
import joblib
joblib.dump(rf, 'Titanic-LR-20220410-5.pkl', compress=3)
'''