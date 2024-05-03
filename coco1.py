
import numpy as np
import pandas as pd


from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

df1=pd.read_excel("C:/Users/Ravi/Downloads/OneDrive/Data Science/esemble techniques/Datasets_ET/Coca_Rating_Ensemble.xlsx")
df1.isnull().sum()
df1.columns
df1.head()
df1.dtypes
df1.shape
df1.info()
df1.drop_duplicates()

  


####
df1['Rating'].unique()
df1['Rating'].value_counts()
df1['Rating']=df1['Rating'].apply(np.floor)
df1['Rating']=df1['Rating'].astype(int)
####
df1['Review']=df1['Review'].astype(int)
#####
df1['Company'].unique()
df1['Company'].replace('Na�ve','Naive',inplace=True)
######
df1.drop(['REF'], axis=1, inplace=True)

#####
# Data cleaning the Bean Type column

df1['Bean_Type'].unique()
df1.Bean_Type.value_counts()
df1['Bean_Type'] = df1['Bean_Type'].fillna('unknown')# filling the null values with unknown

    

df1['Bean_Type'].replace('Forastero (Arriba) ASSS', 'Forastero',inplace=True)
df1['Bean_Type'].replace('Forastero (Arriba) ASS', 'Forastero',inplace=True)
df1['Bean_Type'].replace('Forastero (Arriba)', 'Forastero',inplace=True)
df1['Bean_Type'].replace('Forastero (Nacional)', 'Forastero',inplace=True)
df1['Bean_Type'].replace('Criollo, +','Criollo',inplace=True)
df1['Bean_Type'].replace('Blend-Forastero,Criollo','Blend',inplace=True)
df1['Bean_Type'].replace('Forastero(Arriba, CCN)','Forastero',inplace=True)
df1['Bean_Type'].replace('Forastero (Amelonado)','Forastero',inplace=True)
df1['Bean_Type'].replace('Trinitario, Nacional','Trinitario',inplace=True)
df1['Bean_Type'].replace('Trinitario (Amelonado)','Trinitario',inplace=True)
df1['Bean_Type'].replace('Trinitario, TCGA','Trinitario',inplace=True)
df1['Bean_Type'].replace('Criollo (Amarru)','Criollo',inplace=True)
df1['Bean_Type'].replace('Criollo, Trinitario','Blend',inplace=True)
df1['Bean_Type'].replace('Criollo (Porcelana)','Criollo',inplace=True)
df1['Bean_Type'].replace('Trinitario (85% Criollo)','Blend',inplace=True)
df1['Bean_Type'].replace('Forastero (Catongo)','Forastero',inplace=True)
df1['Bean_Type'].replace('Forastero (Parazinho)','Forastero',inplace=True)
df1['Bean_Type'].replace('Trinitario, Criollo','Blend',inplace=True)
df1['Bean_Type'].replace('Criollo (Ocumare)','Criollo',inplace=True)
df1['Bean_Type'].replace('Criollo (Ocumare 61)','Criollo',inplace=True)
df1['Bean_Type'].replace('Criollo (Ocumare 77)','Criollo',inplace=True)
df1['Bean_Type'].replace('Criollo (Ocumare 67)','Criollo',inplace=True)
df1['Bean_Type'].replace('Criollo (Wild)','Criollo',inplace=True)
df1['Bean_Type'].replace('Trinitario, Forastero','Blend',inplace=True)
df1['Bean_Type'].replace('Trinitario (Scavina)','Trinitario',inplace=True)
df1['Bean_Type'].replace('Criollo, Forastero','Blend',inplace=True)
df1['Bean_Type'].replace('Forastero, Trinitario','Blend',inplace=True)
df1.replace('\xa0','Unkown',inplace=True)
df1['Bean_Type'].replace(np.nan,'Unkown',inplace=True)
df1['Bean_Type'].unique()
#########

df1['Cocoa_Percent']=df1['Cocoa_Percent'].replace("%",'')

########
df1['Company_Location'].value_counts()
df1['Company_Location'].replace({'Domincan Republic':'Dominican Republic','Niacragua':'Nicaragua','Eucador':'Ecuador'},inplace=True)
#######
# Data cleaning the Origin column

df1['Origin'].unique()
#######

df1.replace({'Unkown':np.nan,},inplace=True)
df1.isnull().sum()
#######

# filling the missing values with most frequent values in beanType Country
df1.Bean_Type.fillna(df1.Bean_Type.mode()[0], inplace=True)
df1.Origin.fillna(df1.Origin.mode()[0], inplace=True)
df1.isnull().sum()

import os

# Replace the following path with the desired location on your PC
output_path =  r"C:/Users/Ravi/Downloads/New projects/coca clean.csv"
df1.to_csv(output_path, index=False)
print(f"The DataFrame has been saved to: {os.path.abspath(output_path)}")
###########


plt.figure(figsize=(10,6))

sns.countplot(x = df1['Review'])
plt.plot()

sns.distplot(df1.Cocoa_Percent)
plt.show()
#########

df1.head()
col = ['Company', 'Name','Origin', 'Company_Location', 'Bean_Type']
le = LabelEncoder()
for LE in col:
    df1[LE] = le.fit_transform(df1[LE])
    
####
#dependent and independent variables

X=df1.drop(['Rating'],axis=1)      #independent variable

y=df1['Rating']                    #dependent variable


# Top 5 companies in terms of average ratings
d2 = df1.groupby('Company').aggregate({'Rating':'mean'})
d2 = d2.sort_values('Rating', ascending=False).head(5)
d2 = d2.reset_index()

# Plotting
sns.set()
plt.figure(figsize=(20, 6))
sns.barplot(x='Company', y='Rating', data=d2)
plt.xlabel("\nChocolate Company")
plt.ylabel("Average Rating")
plt.title("Top 5 Companies in terms of Average Ratings \n")
plt.show()

d3 = df1['Origin'].value_counts().sort_values(ascending=False).head(5)
d3 = pd.DataFrame(d3)
d3 = d3.reset_index()
# Plotting
sns.set()
plt.figure(figsize=(10, 6))
sns.barplot(x='index', y='Origin', data=d3)
plt.xlabel("Company_Location")
plt.ylabel("Number of Chocolate Bars")
plt.title("Where does Soma get it's beans from? \n")
plt.show()
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()

X=SS.fit_transform(X)
y=np.array(y)
y=y.reshape(-1,1)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
X_train.shape #(1346, 7)
y_train.shape


#################

from numpy import mean
from numpy import std

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot
import numpy as np


######################################
# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LogisticRegression()))
	level0.append(('knn', KNeighborsClassifier()))
	level0.append(('cart', DecisionTreeClassifier()))
	level0.append(('svm', SVC()))
	level0.append(('bayes', GaussianNB()))
	# define meta learner model
	level1 = LogisticRegression()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores
 

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

###################################


############################# voting

# Accuracy of hard voting
##voting

from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
# Instantiate the learners (classifiers)
learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
learner_3 = svm.SVC(gamma=0.001)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_1),
                           ('Prc', learner_2),
                           ('SVM', learner_3)])

# Fit classifier with the training data
voting.fit(X_train, y_train)

# Predict the most voted class
hard_predictions = voting.predict(X_test)


# Accuracy of hard voting
print('Hard Voting:', accuracy_score(y_test, hard_predictions))

hard_predictions1 = voting.predict(X_train)

print('Hard Voting:', accuracy_score(y_train, hard_predictions1))



#######
#################

# Soft Voting # 
# Instantiate the learners (classifiers)
learner_4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
learner_5 = naive_bayes.GaussianNB()
learner_6 = svm.SVC(gamma = 0.001, probability = True)

# Instantiate the voting classifier
voting = VotingClassifier([('KNN', learner_4),
                           ('NB', learner_5),
                           ('SVM', learner_6)],
                            voting = 'soft')

# Fit classifier with the training data
voting.fit(X_train, y_train)
learner_4.fit(X_train, y_train)
learner_5.fit(X_train, y_train)
learner_6.fit(X_train, y_train)

# Predict the most probable class
soft_predictions = voting.predict(X_test)

# Get the base learner predictions
predictions_4 = learner_4.predict(X_test)
predictions_5 = learner_5.predict(X_test)
predictions_6 = learner_6.predict(X_test)

# Accuracies of base learners
print('L4:', accuracy_score(y_test, predictions_4))
print('L5:', accuracy_score(y_test, predictions_5))
print('L6:', accuracy_score(y_test, predictions_6))

# Accuracy of Soft voting
print('Soft Voting:', accuracy_score(y_test, soft_predictions))


#######################

###############################
################# Bagging
from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score, confusion_matrix


dt = DT(criterion = 'entropy')
bag_clf = BaggingClassifier(base_estimator =dt , n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)

bag_clf.fit(X_train, y_train)



# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(X_test))
accuracy_score(y_test, bag_clf.predict(X_test))

# Evaluation on Training Data
confusion_matrix(y_train, bag_clf.predict(X_train))
accuracy_score(y_train, bag_clf.predict(X_train))


import xgboost as xgb
   
xgb_clf = xgb.XGBRegressor(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)
   
    # n_jobs – Number of parallel threads used to run xgboost.
    # learning_rate (float) – Boosting learning rate (xgb’s “eta”)
    
xgb_clf.fit(X_train, y_train)
    # Evaluation on Testing Data
    # getting error value
from sklearn.metrics import mean_squared_error,r2_score

print("MSE(Test)",mean_squared_error(y_test, xgb_clf.predict(X_test)))
print("R^2(Test)",r2_score(y_test, xgb_clf.predict(X_test)))
    # Evaluation on Training Data\n",
    # getting error value",
print("MSE(Train)",mean_squared_error(y_train, xgb_clf.predict(X_train)))
    # R^2 scuare value\n",
print("R^2(Train)",r2_score(y_train, xgb_clf.predict(X_train)))
    
xgb.plot_importance(xgb_clf);plt.show() 

####################




from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()

boost_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, boost_clf.predict(X_test))
accuracy_score(y_test, boost_clf.predict(X_test))


confusion_matrix(y_train, boost_clf.predict(X_train))
accuracy_score(y_train, boost_clf.predict(X_train))



# Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)
boost_clf2.fit(X_train, y_train)



# Evaluation on Testing Data
confusion_matrix(y_test, boost_clf2.predict(X_test))
accuracy_score(y_test, boost_clf2.predict(X_test))

# Evaluation on Training Data
accuracy_score(y_train, boost_clf2.predict(X_train))

###############################

#########adaboosting########################
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(X_train, y_train)

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(X_test))
accuracy_score(y_test, ada_clf.predict(X_test))

# Evaluation on Training Data
confusion_matrix(y_train, ada_clf.predict(X_train))
accuracy_score(y_train, ada_clf.predict(X_train))




