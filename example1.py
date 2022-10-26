import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree,ensemble,metrics
from PIL import Image
from rule import Rule
from rule_extraction import rule_extract,draw_tree,rules_vote

# fetch dataset
data = pd.read_csv('./dataset/city_day.csv', usecols = ['PM10', 'NO','CO','Benzene','AQI_Bucket'])
# aqi_class = pd.read_csv('./dataset/city_day.csv', usecols = ['AQI_bucket'])

# drop NA records since most Tree algorithm cannot handle
data.dropna(inplace=True)
res  = []

for i in data['AQI_Bucket']:
    if i=="Good":
        res.append(0)
    elif i=="Satisfactory":
        res.append(1)
    elif i=="Moderate":
        res.append(2)
    elif i=="Poor":
        res.append(3)
    elif i=="Very Poor":
        res.append(4)
    elif i=="Severe":
        res.append(5)
    else:
        res.append(-1)

data["aqi_res"]  = res


# # split training/test sets
X_train, X_test, y_train, y_test = train_test_split(data[['PM10', 'NO','CO','Benzene',"AQI_Bucket",'aqi_res']], 
                                                    data.aqi_res, test_size=0.2, random_state=0)

# dataset shape
print(X_train.shape, X_test.shape)
print("Train data: \n", X_train.head(5))
print("Test data: \n", X_test.head(5))

# PM10 mean encoding
X_train.groupby(['PM10'])['aqi_res'].mean()
ordered_labels = X_train.groupby(['PM10'])['aqi_res'].mean().to_dict()
ordered_labels

# Mean Encoding
X_train['PM_ordered'] = X_train.PM10.map(ordered_labels)
X_test['PM_ordered'] = X_test.PM10.map(ordered_labels)


# Sex
X_train.groupby(['NO'])['aqi_res'].mean()
ordered_labels = X_train.groupby(['NO'])['aqi_res'].mean().to_dict()
ordered_labels

# Mean Encoding
X_train['NO_ordered'] = X_train.NO.map(ordered_labels)
X_test['NO_ordered'] = X_test.NO.map(ordered_labels)


X_train_proceeded = X_train[['CO', 'Benzene','PM_ordered','NO_ordered']]
X_test_proceeded = X_test[['CO', 'Benzene','PM_ordered','NO_ordered']]
print(X_train_proceeded.head())

# API refer to http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

model_tree_clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=4)
model_tree_clf.fit(X_train_proceeded,y_train)

# model performance on training set
y_pred = model_tree_clf.predict(X_train_proceeded)
print(metrics.confusion_matrix(y_train,y_pred))

rule, _ = rule_extract(model=model_tree_clf,feature_names=X_train_proceeded.columns)
for i in rule:
    print(i)

# blue node (class=1) denote the node make prediction of class 1
# orange node (class=0) denote the node make prediction of class 0
#  the darker the color, the more purity the node has 
# values refer to the absolute number of labeled samples in that node
# eg, the 1st leaf node [12,7] means that 12 class 0 samples and 7 class 1 samples are in that node

draw_tree(model=model_tree_clf,
          outdir='./images/DecisionTree/',
          feature_names=X_train_proceeded.columns,
          proportion=False,
          class_names=['Good','Satisfactory', 'Moderate', 'Poor', 'Very Poor','Severe'])


# importing Image class from PIL package

 
# creating a object
im = Image.open(r"./images/DecisionTree/DecisionTree.jpeg")
im.show()