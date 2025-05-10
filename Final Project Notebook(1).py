#!/usr/bin/env python
# coding: utf-8

# # Final Project Notebook

# ## Load Libraries and Load Data

# In[2]:


# Importing of critical libaries to support Decision Tree, Regression, and visualization tools.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score


# In[3]:


# Use Pandas to read the Excel file.
df = pd.read_excel('PIRUS_CLEAN_3.xlsx')


# ## Clean and Prepare Data

# Prior to loading the dataset into Anaconda, it was precleaned in Excel, removing features not intended for use.

# In[4]:


# Establishes the selected features and target
selected_features = [
    'Radicalization_Sequence',
    'Radical_Beliefs',
    'Radical_Behaviors',
    'Military',
    'Religious_Background',
    'Education',
    'Age',
    'Sex'
]
target = 'Violent'


# ### Handling of Unknowns

# In[5]:


# Sets the -99, 'unknown values' to NaN and drops.
df_model = df[selected_features + [target]]
df_model = df_model.replace(-99, np.nan).dropna()


# In[6]:


# Assignment of the x and y variable as target and geatures
x = df_model[selected_features]
y = df_model[target]

# Conversion of values to numeric, set NaN values and drop them. Ensure y values only correspond to cleaned x values.
x = x.apply(pd.to_numeric, errors='coerce').dropna()
y = y.loc[x.index]


# ### Normalization

# In[7]:


# Incorporate StandardScaler in an effort to normalize the dataset's features by removing the mean and scaling to unit variance.
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# ## Testing and Training

# In[10]:


# Split into train/test sets.
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.3, random_state=42, stratify=y
)


# In[11]:


# Train the Logistic Regression model.
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(x_train, y_train)
y_pred_log = log_model.predict(x_test)


# In[14]:


# Train the Decision Tree model.
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(x_train, y_train)
y_pred_tree = tree_model.predict(x_test)


# ## Evaluation and Graphical Representations of Model Effectiveness

# In[30]:


# Define  the evaluation  function, print the header name for each model, develop the confusion matrix and heatmap with red coloring with Seaborn, label the x and y axis with matplotlib.
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n--- {model_name} ---")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# In[31]:


# Evaluate both models
evaluate_model(y_test, y_pred_log, "Logistic Regression Model")
evaluate_model(y_test, y_pred_tree, "Decision Tree Model")


# In[38]:


# Plot the Receiver Operating Characteristic Curve Comparison of the Logistic Regression and Decision Tree to see the Area Under Curve values.
log_probs = log_model.predict_proba(x_test)[:, 1]
tree_probs = tree_model.predict_proba(x_test)[:, 1]

fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, color='red', label=f"Logistic Regression (AUC = {roc_auc_score(y_test, log_probs):.2f})")
plt.plot(fpr_tree, tpr_tree, color='orange', label=f"Decision Tree (AUC = {roc_auc_score(y_test, tree_probs):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Pos")
plt.ylabel("True Pos")
plt.title("ROC Curve ")
plt.legend()
plt.grid()
plt.show()


# In[ ]:




