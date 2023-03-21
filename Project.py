import pandas as pd
messages=pd.read_csv('SMSSpamCollection.txt',sep="\t",names=['label','message'])
print(messages.head(2))
print("\n messages\n")
print(messages['message'].head(2))
print("\n label\n")
print(messages['label'].head(2))
print("\n no of records\n",len(messages))

import matplotlib.pyplot as  plt

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',',',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=''.join(review)
    corpus.append(review)
print("corpus")
print(corpus)
from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer(max_features=200)
X=cv.fit_transform(corpus).toarray()
print("\n\n X values\n")
print(X)
y=pd.get_dummies(messages['label'])
print("\ny values\n")
print(y)
y=y.iloc[:,1].values
print("\ny iloc values\n")
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.55,random_state=42)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train,y_train)
y_pred=spam_detect_model.predict(X_test)
print("\n y_pred values\n")
print(y_pred)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
print("\naccuracy\n")
print(acc)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from scipy import stats


#X = np.array(df.drop('Label', axis=1))
#y = np.array(df['Label'])

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,random_state=42)

linear = LinearRegression().fit(X_train, y_train)

print(len(X_test))
print(len(y_test))

coeff = linear.coef_
intercept = linear.intercept_

print('To retrieve the intercept:\n')
print(intercept)

print('For retrieving the slope:\n')
print(coeff )
y_pred =linear.predict(X_test)
print('y_test:\n')
print(y_test)
print('y_pred:\n')
print(y_pred)
df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df1)

df2 = df1.head(25)
df2.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('\n Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('\n Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('\n Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
### Assume y is the actual value and f is the predicted values
r2 = linear.score(X_test,y_test)
print('r2 score for a model which predicts mean value always is', r2)
t2, p2 = stats.ttest_ind(y_test,y_pred)
print("\n t-test value = " + str(t2))
print("\n p-test value = " + str(p2))
## F test-value calculation
var_x = y_test.var()
var_y = y_pred.var()
f=var_y/var_x
print('\n F-Test value =',f)

plt.scatter(y_test,y_pred,color='red')
plt.plot(X_train, linear.predict(X_train), color='blue')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

# Plot the residuals after fitting a linear model

y_pred1 =np.round(linear.predict(X_test))
print('confusion_matrix')
print(confusion_matrix(y_pred1,y_test))
print('accuracy_score')
print(accuracy_score(y_pred1,y_test))
print('classification_report')
print(classification_report(y_pred1,y_test))


from xgboost import XGBClassifier
xgbcl = XGBClassifier()
xgbcl.fit(X_train , y_train)
y_pred3 = xgbcl.predict(X_test)
print ("Accuracy of Xgboost= ", metrics.accuracy_score(y_test,y_pred3))


from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 200, random_state = 42)
RF_model.fit(X_train , y_train)
y_pred4= RF_model.predict(X_test )
print ("Accuracy of Random forest= ", metrics.accuracy_score(y_test,y_pred4))


from lightgbm import LGBMClassifier
lgb_model = LGBMClassifier()
lgb_model.fit(X_train , y_train)
y_pred5 = lgb_model.predict(X_test)
print ("\n Accuracy of Light GBM= ", metrics.accuracy_score(y_test,y_pred5))

from sklearn.ensemble import GradientBoostingClassifier
gb_model=GradientBoostingClassifier()
gb_model.fit(X_train , y_train)
y_pred6 = gb_model.predict(X_test)
print ("\n Accuracy of  GBM= ", metrics.accuracy_score(y_test,y_pred6))

from sklearn.neighbors import KNeighborsClassifier
kn_model=KNeighborsClassifier(n_neighbors=3)
kn_model.fit(X_train , y_train)
y_pred7 = kn_model.predict(X_test)
print ("\n Accuracy of  KNeighborsClassifier= ", metrics.accuracy_score(y_test,y_pred7))

from sklearn.linear_model import LogisticRegression as lgrClassifier

lgr = lgrClassifier(C = 100)
lgr.fit(X_train , y_train)
y_pred1 = lgr.predict(X_test)

print ("Accuracy of Logistic regresion= ", metrics.accuracy_score(y_test,y_pred1))

from sklearn.metrics import classification_report 
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
    HistGradientBoostingClassifier
)

bagging = BaggingClassifier(n_jobs=-1)
extraTrees = ExtraTreesClassifier(max_depth=10, n_jobs=-1)
randomForest = RandomForestClassifier(n_jobs=-1)
histGradientBoosting = HistGradientBoostingClassifier()
XGB = XGBClassifier(n_jobs=-1)

model = StackingClassifier([
    ('bagging', bagging),
    ('extraTress', extraTrees),
    ('randomforest', randomForest),
    ('histGradientBoosting', histGradientBoosting),
    ('XGB', XGB)
], n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print ("\n Accuracy of  StackingClassifier= ", metrics.accuracy_score(y_test,y_pred))

from sklearn import tree
DTC_model = tree.DecisionTreeClassifier()
DTC_model.fit(X_train , y_train)
y_pred2=DTC_model.predict(X_test)
print ("Accuracy of Decison tree= ", metrics.accuracy_score(y_test,y_pred2))






