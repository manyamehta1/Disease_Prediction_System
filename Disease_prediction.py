#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('HTML', '', '\n<html>\n<head>\n  <title>Disease Prediction Using Machine Learning</title>\n  <style>\n    /* Define the styling for the text */\n    .stylish-text {\n      font-family: Arial, sans-serif;\n      font-size: 24px;\n      color: #ffffff;\n      background-color: #336699;\n      padding: 10px;\n      border-radius: 5px;\n      text-align: center;\n      text-shadow: 2px 2px #000000;\n      animation: colorChange 2s infinite alternate;\n    }\n\n    @keyframes colorChange {\n      0% {\n        background-color: #336699;\n      }\n      50% {\n        background-color: #993366;\n      }\n      100% {\n        background-color: #669933;\n      }\n    }\n  </style>\n</head>\n<body>\n  <div class="stylish-text">COVID SYMPTOM CHECKER USING MACHINE LEARNING</div>\n</body>\n</html>')


# In[43]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('covid_dataset.csv')
data.info()


# In[15]:


# Checking the levels for categorical features

def show(data):
  for i in data.columns[1:]:
    print("Feature: {} with {} Levels".format(i,data[i].unique()))

show(data)


# In[16]:


data.isnull().sum()


# In[17]:


data.Covid19.value_counts()


# In[18]:


sns.countplot(x="Covid19", data=data, palette="bwr")
plt.show()


# In[20]:


accuracies = {}


# In[21]:


countNoDisease = len(data[data.Covid19 == 0])
countHaveDisease = len(data[data.Covid19 == 1])
print("Percentage of not having Covid : {:.2f}%".format((countNoDisease / (len(data.Covid19))*100)))
print("Percentage of having Covid : {:.2f}%".format((countHaveDisease / (len(data.Covid19))*100)))


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

data = pd.read_csv('covid_dataset.csv')

# Separate the features (X) and the target variable (y)
X = data.drop('Covid19', axis=1)
y = data['Covid19']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
rf_classifier = RandomForestClassifier(n_estimators=20,random_state=12,max_depth=5)
rf_classifier.fit(X_train, y_train)
rf_predicted=rf_classifier.predict(X_test)
rf_conf_matrix=confusion_matrix(y_test,rf_predicted)
print("Confusion Matrix")
print(rf_conf_matrix)

# Predict the target variable for the test data
y_pred = rf_classifier.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred)
accuracies['Random Forest']=accuracy_rf;
print("Random Forest Accuracy :- ", accuracy_rf)


# In[26]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('covid_dataset.csv')

# Separate the features (X) and the target variable (y)
X = data.drop('Covid19', axis=1)
y = data['Covid19']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Feature importance
feature_importance = rf_classifier.feature_importances_

# Sort feature importance in descending order
sorted_indices = feature_importance.argsort()[::-1]
sorted_feature_importance = feature_importance[sorted_indices]
sorted_feature_names = X.columns[sorted_indices]

# Select top k features
k = 10  # Change k to the desired number of top features
top_k_features = sorted_feature_names[:k]

# Update X with selected features
X_selected = X[top_k_features]

# Split the selected features into training and testing sets
X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Initialize a new random forest classifier
rf_classifier_selected = RandomForestClassifier()

# Train the model on the training data with selected features
rf_classifier_selected.fit(X_train_selected, y_train)

# Predict the target variable for the test data with selected features
y_pred_selected = rf_classifier_selected.predict(X_test_selected)

# Calculate the accuracy of the model with selected features
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy with selected features: ", accuracy_selected)


# In[28]:


from sklearn.tree import DecisionTreeClassifier
# Load the dataset from a CSV file
data = pd.read_csv('covid_dataset.csv')

# Separate the features (X) and the target variable (y)
X = data.drop('Covid19', axis=1)
y = data['Covid19']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)


dt_classifier = DecisionTreeClassifier()

# Train the model on the training data
dt_classifier.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = dt_classifier.predict(X_test)

dt_classifier = RandomForestClassifier(n_estimators=20,random_state=12,max_depth=5)
dt_classifier.fit(X_train, y_train)
dt_predicted=dt_classifier.predict(X_test)
dt_conf_matrix=confusion_matrix(y_test,dt_predicted)
print("Confusion Matrix")
print(dt_conf_matrix)
accuracy_dt = accuracy_score(y_test, y_pred)
accuracies['Decision Tree']=accuracy_dt;
print("Accuracy :- ", accuracy_dt)


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset from a CSV file
data = pd.read_csv('covid_dataset.csv')

# Separate the features (X) and the target variable (y)
X = data.drop('Covid19', axis=1)
y = data['Covid19']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a decision tree classifier
dt_classifier = DecisionTreeClassifier()

# Train the model on the training data
dt_classifier.fit(X_train, y_train)

# Feature importance
feature_importance = dt_classifier.feature_importances_

# Sort feature importance in descending order
sorted_indices = feature_importance.argsort()[::-1]
sorted_feature_importance = feature_importance[sorted_indices]
sorted_feature_names = X.columns[sorted_indices]

# Select top k features
k = 10  # Change k to the desired number of top features
top_k_features = sorted_feature_names[:k]

# Update X with selected features
X_selected = X[top_k_features]

# Split the selected features into training and testing sets
X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=38)

# Initialize a new decision tree classifier
dt_classifier_selected = DecisionTreeClassifier()

# Train the model on the training data with selected features
dt_classifier_selected.fit(X_train_selected, y_train)

# Predict the target variable for the test data with selected features
y_pred_selected = dt_classifier_selected.predict(X_test_selected)

# Calculate the accuracy of the model with selected features
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy with selected features: ", accuracy_selected)


# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset from a CSV file
data = pd.read_csv('covid_dataset.csv')

# Separate the features (X) and the target variable (y)
X = data.drop('Covid19', axis=1)
y = data['Covid19']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

# Initialize different Naive Bayes models
naive_bayes_models = [
    ('GaussianNB', GaussianNB()),
    # Add more variations of Naive Bayes models here if desired
]


# Train and evaluate each Naive Bayes model
accuracy_scores = []
for model_name, nb_classifier in naive_bayes_models:
    nb_classifier.fit(X_train, y_train)
    y_pred = nb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append((model_name, accuracy))

# nb_classifier = GaussianNB(n_estimators=20,random_state=12,max_depth=5)
nb_classifier.fit(X_train, y_train)
nb_predicted=nb_classifier.predict(X_test)
nb_conf_matrix=confusion_matrix(y_test,nb_predicted)
print("Confusion Matrix")
print(nb_conf_matrix)
accuracy_nb = accuracy_score(y_test, y_pred)
accuracies['Naive Bayes']=accuracy_nb;
print("Accuracy :- ", accuracy_nb)


# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Load the dataset from a CSV file
data = pd.read_csv('covid_dataset.csv')

# Separate the features (X) and the target variable (y)
X = data.drop('Covid19', axis=1)
y = data['Covid19']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

# Select top k features using SelectKBest and chi-squared test
k = 10  # Change k to the desired number of top features
selector = SelectKBest(score_func=chi2, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]

# Initialize a Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the model on the training data with selected features
nb_classifier.fit(X_train_selected, y_train)

# Predict the target variable for the test data with selected features
y_pred_selected = nb_classifier.predict(X_test_selected)

# Calculate the accuracy of the model with selected features
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Accuracy with selected features: ", accuracy_selected)


# In[35]:


import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset from a CSV file
data = pd.read_csv('covid_dataset.csv')

# Separate the features (X) and the target variable (y)
X = data.drop('Covid19', axis=1)
y = data['Covid19']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize an SVM classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
svm_predicted=svm_classifier.predict(X_test)
svm_conf_matrix=confusion_matrix(y_test,svm_predicted)
print("Confusion Matrix")
print(svm_conf_matrix)

# Train the model on the training data
svm_classifier.fit(X_train_scaled, y_train)

# Predict the target variable for the test data
y_pred = svm_classifier.predict(X_test_scaled)


accuracy_svm = accuracy_sco.0aZSWqqre(y_test, y_pred)
accuracies['Support Vector Machine']=accuracy_svm;
print("Accuracy :- ", accuracy_svm)


# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
import seaborn as sns

# Load the dataset from a CSV file
data = pd.read_csv('covid_dataset.csv')

# Separate the features (X) and the target variable (y)
X = data.drop('Covid19', axis=1)
y = data['Covid19']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

# Select top k features using SelectKBest and chi-squared test
k = 10  # Change k to the desired number of top features
selector = SelectKBest(score_func=chi2, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]

# Initialize a Gaussian Naive Bayes classifier
nb_classifier = SVC()

# Train the model on the training data with selected features
nb_classifier.fit(X_train_selected, y_train)

# Predict the target variable for the test data with selected features
y_pred_selected = nb_classifier.predict(X_test_selected)

# Calculate the accuracy of the model with selected features
accuracy_selected = accuracy_score(y_test, y_pred_selected)
accuracies['Support Vector Machine']=accuracy_selected;
print("Accuracy with selected features: ", accuracy_selected)


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

colors = ["purple", "green", "orange", "magenta"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()


# In[38]:


fig = plt.gcf()
fig.set_size_inches(20, 10)
sns.heatmap(data.corr(), annot = True)
plt.show()


# In[39]:


model_ev = pd.DataFrame({'Models' :['Support Vector Machine', 'Naive Bayes', 'Decision Tree', 'Random Forest'],
                         'Accuracy': [accuracy_svm*100, accuracy_nb*100, accuracy_dt*100, accuracy_rf*100]
                         })


# In[40]:


model_ev


# In[41]:


plt.bar(model_ev['Models'], model_ev['Accuracy'])

# Set the title and labels
plt.title('Model Evaluation')
plt.xlabel('Models')
plt.ylabel('Accuracy')

# Rotate the x-axis labels if needed
plt.xticks(rotation=45)

# Display the plot
plt.show()

