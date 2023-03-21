import warnings
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.preprocessing import LabelEncoder as le
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import RobustScaler as rbScaler 
from sklearn.linear_model import LogisticRegression as lgrClassifier
from sklearn import metrics
warnings.filterwarnings('ignore')
df = pd.read_csv('C:\\Users/Admin/Desktop/Crditscore/train.csv',low_memory=False)
num_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
       'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
       'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
       'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
       'Total_EMI_per_month', 'Amount_invested_monthly',
       'Monthly_Balance', 'Credit_History_Age']

categorical_cols = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount',
       'Payment_Behaviour', 'Credit_Score']
irrelavent_coulumns = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN']
df.drop(columns=irrelavent_coulumns, inplace=True, axis=1)
df = df.applymap(
    lambda x: x if x is np.NaN or not \
        isinstance(x, str) else str(x).strip('_')).replace(
            ['', 'nan', '!@9#%8', '#F%$D@*&8'], np.NaN
        )
df.Age = df.Age.astype(int)
df.Annual_Income = df.Annual_Income.astype(float)
df.Num_of_Loan = df.Num_of_Loan.astype(int)
df.Num_of_Delayed_Payment = df.Num_of_Delayed_Payment.astype(float)
df.Changed_Credit_Limit = df.Changed_Credit_Limit.astype(float)
df.Outstanding_Debt = df.Outstanding_Debt.astype(float)
df.Amount_invested_monthly = df.Amount_invested_monthly.astype(float)
df.Monthly_Balance = df.Monthly_Balance.astype(float)
def take_years(x):  
    if x is not None:
        return str(x).strip()[0:2]

df.Credit_History_Age=df.Credit_History_Age.apply(take_years)
df['Credit_History_Age'] = df['Credit_History_Age'].replace({'na': np.NaN})
df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace({'NM': 'No'})
def remove_outlier(df):
    low = .05
    high = .95
    quant_df = df.quantile([low, high])
    #print(quant_df)
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
    return df

df = remove_outlier(df)
Occupation_le = le()
Type_of_Loan_le = le()
Credit_Mix_le = le()
Credit_History_Age_le = le()
Payment_of_Min_Amount_le = le()
Payment_Behaviour_le = le()
Credit_Score_le = le()

df['Occupation'] = Occupation_le.fit_transform(df['Occupation'])
df['Type_of_Loan'] = Type_of_Loan_le.fit_transform(df['Type_of_Loan'])
df['Credit_Mix'] = Credit_Mix_le.fit_transform(df['Credit_Mix'])
df['Credit_History_Age'] = Credit_History_Age_le.fit_transform(df['Credit_History_Age'])
df['Payment_of_Min_Amount'] = Payment_of_Min_Amount_le.fit_transform(df['Payment_of_Min_Amount'])
df['Payment_Behaviour'] = Payment_Behaviour_le.fit_transform(df['Payment_Behaviour'])
df['Credit_Score'] = Credit_Score_le.fit_transform(df['Credit_Score'])
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', ascending=False, inplace=True)
missing_value_df
mdf = df[
    ['Credit_Score','Changed_Credit_Limit',
      'Payment_of_Min_Amount', 'Credit_Mix',
      'Delay_from_due_date', 'Annual_Income',
      'Age', 'Monthly_Balance', 'Outstanding_Debt',
      'Payment_Behaviour', 'Credit_History_Age',
      'Num_Bank_Accounts'
    ]
]
x = mdf.drop(['Credit_Score'] , axis = 1).values
y = mdf['Credit_Score' ].values
from imblearn.over_sampling import SMOTE
rus = SMOTE(sampling_strategy='auto')
X_data_rus, y_data_rus = rus.fit_resample(x, y)
#y_data_rus.value_count(normalize=True)
x_train, x_test, y_train, y_test = train_test_split(X_data_rus, y_data_rus, test_size=0.3, random_state=42,stratify=y_data_rus)
#x_train , x_test , y_train , y_test = train_test_split(x,y , test_size= 0.2 , random_state=50)
#print([x_train.shape, y_train.shape, x_test.shape, y_test.shape])
scalar = PowerTransformer(method='yeo-johnson', standardize=True).fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)
'''# Data Scaling using Robust Scaler
ro_scaler = rbScaler()
x_train = ro_scaler.fit_transform(x_train)
x_test = ro_scaler.fit_transform(x_test)
[x_train.shape, x_test.shape]'''

# logistic Regression
lgr = lgrClassifier(C = 100)
lgr.fit(x_train , y_train)
y_pred1 = lgr.predict(x_test)

print ("Accuracy of Logistic regresion= ", metrics.accuracy_score(y_test,y_pred1))

DTC_model = tree.DecisionTreeClassifier()
DTC_model.fit(x_train , y_train)
y_pred2=DTC_model.predict(x_test)
print ("Accuracy of Decison tree= ", metrics.accuracy_score(y_test,y_pred2))

from xgboost import XGBClassifier
xgbcl = XGBClassifier()
xgbcl.fit(x_train , y_train)
y_pred3 = xgbcl.predict(x_test)
print ("Accuracy of Xgboost= ", metrics.accuracy_score(y_test,y_pred3))

from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 200, random_state = 42)
RF_model.fit(x_train , y_train)
y_pred4= RF_model.predict(x_test )
print ("Accuracy of Random forest= ", metrics.accuracy_score(y_test,y_pred4))

from lightgbm import LGBMClassifier
lgb_model = LGBMClassifier()
lgb_model.fit(x_train , y_train)
y_pred5 = lgb_model.predict(x_test)
print ("Accuracy of Light GBM= ", metrics.accuracy_score(y_test,y_pred5))

gb_model=GradientBoostingClassifier()
gb_model.fit(x_train , y_train)
y_pred6 = gb_model.predict(x_test)
print ("Accuracy of  GBM= ", metrics.accuracy_score(y_test,y_pred6))

kn_model=KNeighborsClassifier(n_neighbors=3)
kn_model.fit(x_train , y_train)
y_pred7 = kn_model.predict(x_test)
print ("Accuracy of  KNeighborsClassifier= ", metrics.accuracy_score(y_test,y_pred7))
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
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print ("Accuracy of  StackingClassifier= ", metrics.accuracy_score(y_test,y_pred))

'''Output:
>>> 
============== RESTART: C:/Users/Admin/Desktop/Crditscore/pgr3.py ==============
Accuracy of Logistic regresion=  0.6591431274496553
Accuracy of Decison tree=  0.7921340721719151
Accuracy of Xgboost=  0.8716042708474118
Accuracy of Random forest=  0.8680902824706042
Accuracy of Light GBM=  0.8530882551696175
Accuracy of  GBM=  0.8051087984862819
Accuracy of  KNeighborsClassifier=  0.8056494120827139
Accuracy of  StackingClassifier=  0.8841735369644547
>>> 





















Accuracy of random forest=  0.7807957153787299
Accuracy of Light GBM=  0.7631981637337414
Accuracy of  GBM=  0.7547819433817904
Accuracy of  KNeighborsClassifier=  0.7276205049732212'''
