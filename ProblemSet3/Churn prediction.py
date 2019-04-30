#!/usr/bin/env python
# coding: utf-8

# In[16]:


get_ipython().run_line_magic('pylab', 'inline --no-import-all')

from scipy import interp

import pandas as pd
from pandas.api.types import CategoricalDtype

import sklearn
from sklearn import ensemble, model_selection, metrics
import xgboost
from xgboost import XGBClassifier

plt.rcParams['figure.figsize'] = (12, 8)


# First we define some functions to convert data from the csv to Python types.

# ## Data loading and preprocessing

# In[6]:


def convert_bool(col, true_str="Yes", false_str="No"):
    """
    Convert string to boolean values.
    """
    if col ==  true_str: #check for nan
        return 1
    elif col == false_str:
        return 0
    else:
        raise ValueError(col)


def convert_float(col):
    """
    Convert floating point values. If it is not possible, it returns 0.
    
    This is useful for the TotalCharges columns, since it contains
    empty cells when it should be 0.
    """
    try:
        return float(col)
    except ValueError:
        return 0


# In[111]:


categorical_no_internet = CategoricalDtype(categories=["No internet service", "No", "Yes"])

df = pd.read_csv(
    "Telco-Customer-Churn.csv", 
    dtype={
        "MultipleLines": CategoricalDtype(categories=["No phone service", "No", "Yes"]),
        'InternetService': 'category',
        'OnlineSecurity': categorical_no_internet,
        'OnlineBackup': categorical_no_internet,
        'DeviceProtection': categorical_no_internet,
        'TechSupport': categorical_no_internet,
        'StreamingTV': categorical_no_internet,
        'StreamingMovies': categorical_no_internet,
        'Contract': 'category',
    #     'PaperlessBilling': 'category',
        'PaymentMethod': 'category',
    },
    converters={
        'gender': lambda x: convert_bool(x, 'Male', "Female"),
        'Partner': convert_bool,
        'Dependents': convert_bool,
        'PhoneService': convert_bool,
        'PaperlessBilling': convert_bool,
        'TotalCharges': convert_float,
        'Churn': convert_bool
    })

# TotalCharges is related to tenure and MonthlyCharges (TotalCharges is approx = to tenure*MonthlyCharges)
# Instead of storing TotalCharges, we just store the deviation between TotalCharges and tenure*MonthlyCharges.
df["discount"] = (df["tenure"] * df["MonthlyCharges"] - df["TotalCharges"])/(np.maximum(df["tenure"], 1))

y = df["Churn"].values
df = df.drop(labels=["customerID", "Churn", "TotalCharges"], axis=1)


# In[112]:


df.dtypes


# We convert categorical fields to multiple binary fields

# In[113]:


df_nocat = pd.get_dummies(df, drop_first=True)
df_nocat.dtypes


# Get values in NumPy's array format

# In[114]:


x = df_nocat.values


# Variables x and y contain the features and the class to predict, respectively.

# ## Classification
# 
# Since y is a boolean array, we will use classifiers to predict y given x.
# 
# We will use two different classifiers: RandomForestClassifier from sklearn and the boosting classifier from XGBoost.

# In[162]:


# We use 1000 estimators (trees)
clsf_forest = ensemble.RandomForestClassifier(n_estimators=1000)
clsf_boost = XGBClassifier(n_estimators=1000)


# ### Cross-validation

# In[135]:


def cv_roc(clsf, x, y, n_splits=4):
    
    results = {'tpr': [], 'fpr': [], 'average_precision': [], "roc_auc": [], "accuracy": []}
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    cv = model_selection.StratifiedKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(x, y):
        train_x, train_y = x[train_idx], y[train_idx]
        test_x, test_y = x[test_idx], y[test_idx]

        clsf.fit(train_x, train_y)
        
        y_pred = clsf.predict_proba(test_x)[:, 1]
        
        fpr, tpr, _ = metrics.roc_curve(test_y, y_pred)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        
        results['fpr'].append(fpr)
        results['tpr'].append(tpr)
        results['average_precision'].append(metrics.average_precision_score(test_y, y_pred))
        results['roc_auc'].append(metrics.roc_auc_score(test_y, y_pred))
        
        results['accuracy'].append(metrics.accuracy_score(test_y, y_pred>0.5))
    
    mean_tpr /= n_splits
    mean_tpr[-1] = 1.0
    
    return {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        **{k: np.array(v) for k, v in results.items()}
    }


# In[136]:


results_forest = cv_roc(clsf_forest, x, y)


# In[137]:


plt.figure(figsize=(9, 9))
plt.plot(results_forest['mean_fpr'], results_forest['mean_tpr'],
         color='black', linewidth=4, label="Mean ROC curve")

for i, (fpr, tpr) in enumerate(zip(results_forest['fpr'], results_forest['tpr'])):
    plt.plot(fpr, tpr, linewidth=2, linestyle='--', label="ROC curve of fold {}".format(i))

plt.legend(loc='lower right', fontsize=14)
plt.xlabel("False positive rate", fontsize=16)
plt.ylabel("True positive rate", fontsize=16)
plt.grid()

plt.title("Random Forest", fontsize=18)


# In[138]:


results_boost = cv_roc(clsf_boost, x, y)


# In[139]:


plt.figure(figsize=(9, 9))
plt.plot(results_boost['mean_fpr'], results_boost['mean_tpr'],
         color='black', linewidth=4, label="Mean ROC curve")

for i, (fpr, tpr) in enumerate(zip(results_boost['fpr'], results_boost['tpr'])):
    plt.plot(fpr, tpr, linewidth=2, linestyle='--', label="ROC curve of fold {}".format(i))

plt.legend(loc='lower right', fontsize=14)
plt.xlabel("False positive rate", fontsize=16)
plt.ylabel("True positive rate", fontsize=16)
plt.grid()

plt.title("XGBoost", fontsize=18)


# In[140]:


plt.figure(figsize=(9, 9))

auc_forest = metrics.auc(results_forest['mean_fpr'], results_forest['mean_tpr'])
auc_boost = metrics.auc(results_boost['mean_fpr'], results_boost['mean_tpr'])

plt.plot(results_forest['mean_fpr'], results_forest['mean_tpr'],
         color='C1', linewidth=4, label="Random forest (AUC={:.2g})".format(auc_forest))
plt.plot(results_boost['mean_fpr'], results_boost['mean_tpr'],
         color='C2', linewidth=4, label="XGBoost (AUC={:.2g})".format(auc_boost))
plt.grid()

plt.legend(loc='lower right', fontsize=14)
plt.xlabel("False positive rate", fontsize=16)
plt.ylabel("True positive rate", fontsize=16)

plt.title("Random forest vs. XGBoost", fontsize=18)


# The average accuracy of both models is:

# In[158]:


acc = results_boost["accuracy"]
print("XGBoost accuracy: {:.2f}%(stddev {:.2f})".format(acc.mean() * 100, acc.std()))


# In[159]:


acc = results_forest["accuracy"]
print("Forest accuracy: {:.2f}%(stddev {:.2f})".format(acc.mean() * 100, acc.std()))


# There are two interesting conclusions from these results:
# 1. The performance of different folds is very similar. This suggests that the dataset is large enough and that we can work with a single training/testing split instead of using cross-validation.
# 2. The difference in performance of the random forest and XGBoost is negligible. 
# 
# In the following discussion, we will evaluate the feature importance estimated by the random forest for the following split:

# In[153]:


train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y, stratify=y)


# We look at the feature importances computed by the random forest to understand which factors are more relevant for churn prediction.

# In[154]:


clsf_forest.fit(train_x, train_y)


# In[155]:


plt.figure(figsize=(12, 8))
plt.stem(clsf_forest.feature_importances_)
plt.xlabel("Feature index", fontsize=16)
plt.ylabel("Importance", fontsize=16)
plt.grid()


# We sort the columns of the dataset by the importance according to the random forest:

# In[156]:


indices = np.argsort(clsf_forest.feature_importances_)[::-1]
df_nocat.columns.values[indices]


# The most important factor for churn prediction is 'tenure', followed by MonthlyCharges.
# 
# We plot the histogram of tenure for customers with churn="no" and churn="yes" separately:

# In[157]:


_ = plt.hist([x[y==0, 4], x[y==1, 4]], 50, density=True)
plt.xlabel("Tenure", fontsize=18)
plt.grid()


# As the histogram shows, customers with smaller tenure are more likely to churn than long-term customers.

# In[167]:


_ = plt.hist([x[y==0, 7], x[y==1, 7]], 50, density=True)
plt.xlabel("Monthly charges", fontsize=18)
plt.grid()


# As expected, customers with higher monthly charges are more likely to churn.
