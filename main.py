import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('data/loan_data.csv')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.ensemble import RandomForestClassifier

# Display some basic information
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Summary statistics for numerical columns
print(data.describe())







# Define the correct order of education levels
education_order = [
    'Highschool',
    'Associate',
    'Bachelor',
    'Master',
    'Doctorate'
]

# Convert 'person_education' to a categorical type with the specified order for education levels

data['person_education'] = pd.Categorical(
    data['person_education'],
    categories=education_order,
    ordered=True
)

# Recreate the pivot table
pivot = data.pivot_table(values='person_income', index='person_gender', columns='person_education', aggfunc='mean')

# Plot the heatmap with the order by education level from lowest to the higghest
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap='YlGnBu')
plt.title('Average Income by Gender and Education Level (Ordered)')
plt.xlabel('Education Level')
plt.ylabel('Gender')
plt.show()

