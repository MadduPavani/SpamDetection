#https://vitalflux.com/text-classification-bag-of-words-model-python-sklearn/
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
## Create sample set of documents
docs = np.array(['Mirabai has won a silver medal in weight lifting in Tokyo olympics 2021',
                 'Sindhu has won a bronze medal in badminton in Tokyo olympics',
                 'Indian hockey team is in top four team in Tokyo olympics 2021 after 40 years',
                 'Mirabai has won a silver medal in weight lifting in Tokyo olympics 2022'])
# Fit the bag-of-words model
bag = vectorizer.fit_transform(docs)
# Get unique words / tokens found in all the documents. The unique words / tokens represents
# the features
print(vectorizer.get_feature_names())
# Associate the indices with each unique word
print(vectorizer.vocabulary_)
# Print the numerical feature vector
print(bag.toarray())
# Creating training data set from bag-of-words  and dummy label
X = bag.toarray()
y = np.array([1,0,0,1])

#Step 3: Create a model and fit it
model = LinearRegression()
model.fit(X, y)
#Step 4: Get results model.coef_ ,model.intercept_
r_sq = model.score(X, y)
print('\n model score R_sq:=', model.intercept_)
print('\n intercepting value C:=', model.intercept_)
print('\n slope M:=', model.coef_)


## Cross Checking with the internal scipy function
print('\n\n BEFORE PREDECTION:\n\n')
t2, p2 = stats.ttest_ind(X,y)
print("\n t-test value = " + str(t2))
print("\n p-test value = " + str(p2))
## F test-value calculation
var_x = X.var()
var_y = y.var()
f=var_x/var_y
print('\n F-Test value=',f)


#Step 5: Predict response
print('\n intial values of y',y)
y_pred1 = model.predict(X)
print('\n\n\n predicted response at first1:\n', y_pred1)

#with formulae
y_pred = model.intercept_ + model.coef_ * X
print('predicted response after fitted model:\n', y_pred)

## Cross Checking with the internal scipy function
print('\n\n AFTER PREDECTION:\n\n')
t2, p2 = stats.ttest_ind(X,y_pred1)
print("\n t-test value = " + str(t2))
print("\n p-test value = " + str(p2))
## F test-value calculation
var_x = X.var()
var_y = y_pred1.var()
f=var_x/var_y
print('\n F-Test value=',f)

#with formulae
print('based on formula')
p=input('enter value to predict:')
q= model.intercept_ + model.coef_ * float(p)
print('predicted value:', q)






