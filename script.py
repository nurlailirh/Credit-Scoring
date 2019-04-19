##Load library
import pandas as pd
import numpy as py
import seaborn as sns
import matplotlib.pyplot as plt

##Load dataset
data = pd.read_excel(dataset.xlsx')

##Remove unnecessary columns
data.drop(['nilai_likuidasi'], axis = 1, inplace = True)
data.drop(['pemasukan_total'], axis = 1, inplace = True)

##Handling missing values
#filling with mean
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values="NaN", strategy="mean" )
data["umur"]=imp.fit_transform(data[["umur"]]).ravel()
data["angsuran"]=imp.fit_transform(data[["angsuran"]]).ravel()
#Categorical missing values
import numpy
from sklearn.base import TransformerMixin
class SeriesImputer(TransformerMixin):
def __init__(self):
"""Impute missing values.
If the Series is of dtype Object, then impute with the
most frequent object.
If the Series is not of dtype Object, then impute with
the mean.
"""
def fit(self, X, y=None):
if X.dtype == numpy.dtype('O'): self.fill =
X.value_counts().index[0]
else : self.fill = X.mean()
return self
def transform(self, X, y=None):
return X.fillna(self.fill)
a = SeriesImputer() # Initialize the imputer
#filling with modus for edu code
a.fit(data["pendidikan"]) # Fit the imputer
s2 = a.transform(data["pendidikan"]) # Get a new series
data["pendidikan"]=s2
#filling with modus for job code
a.fit(data["pekerjaan"]) # Fit the imputer
s2 = a.transform(data["pekerjaan"]) # Get a new series
data["pekerjaan"]=s2
#filling with modus for marriage code
a.fit(data["pernikahan"]) # Fit the imputer
s2 = a.transform(data["pernikahan"]) # Get a new series
data["pernikahan"]=s2
#filling with modus for gender code
a.fit(data["jenis_kelamin"]) # Fit the imputer
s2 = a.transform(data["jenis_kelamin"]) # Get a new series
data["jenis_kelamin"]=s2

##One-hot-encoding
#set predictors (x) and target (y)
X_features = list(data.columns)
X_features.remove('Kategori')
Y_feature=data['Kategori'] #variabel output
from sklearn.preprocessing import LabelEncoder
df_complete = pd.get_dummies( data[X_features] )
list(df_complete)
#set predictors and target
predictors = df_complete
target = Y_feature
#label encoding
def dummyEncode(df):
columnsToEncode =
list(df.select_dtypes(include=['category','object']))
le = LabelEncoder()
for feature in columnsToEncode:
try:
df[feature] = le.fit_transform(df[feature])
except:
print('Error encoding '+feature)
return df
dummyEncode(predictors)

##Split training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =
train_test_split(predictors,target,stratify=target,test_size=0.2)

##Sampling
from imblearn.over_sampling import RandomOverSampler
sampletech = RandomOverSampler(random_state=0)
X_resampled, y_resampled = sampletech.fit_sample(X_train,
y_train)
from collections import Counter
print('Resampled dataset shape {}'.format(Counter(y_resampled)))

##Normalization
from sklearn.preprocessing import StandardScaler
#Normalize data
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
# Now apply the transformations to the data:
X_trainf = scaler.transform(X_resampled)
X_testf = scaler.transform(X_test)
y_train = y_resampled

##Import library
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer

##Deciding the model configuration
#make scorer for grid search
def tp(y_true, y_pred):
error= confusion_matrix(y_true,
y_pred)[0,0]/(confusion_matrix(y_true, y_pred)[0,0] +
confusion_matrix(y_true, y_pred)[0,1])
return error
specificity = make_scorer(tp, greater_is_better=True)
scoring={'Accuracy': make_scorer(accuracy_score),'Precision':
make_scorer(precision_score),'Recall':
make_scorer(recall_score),'Specificity': specificity}
#Grid search parameters
parameters = {'activation':('logistic','tanh',
'relu'),'alpha':(0.00001,0.0001,0.001,0.01,0.1,1)}
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
#Model configuration
model =
GridSearchCV(MLPClassifier(hidden_layer_sizes=(607,121),max_iter=
500,solver='adam',random_state=42,verbose=True),
param_grid=parameters,
scoring=scoring,
refit=False,return_train_score=False)
$Fit the model
model.fit(X_trainf,y_train)
#Results
results = model.cv_results_

##Testing
#Model configuration
model=MLPClassifier(hidden_layer_sizes=(607,121),activation='tanh
',max_iter=500,solver='adam',verbose=True,alpha=0.0001)
#Fit model
model.fit(X_trainf,y_train)
loss_values = model.loss_curve_
#Test the model
y_pred = model.predict(X_testf)
#Print final result
print(confusion_matrix(y_test,y_pred))