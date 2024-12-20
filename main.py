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




# create age groups for decades
data['age_group'] = pd.cut(
    data['person_age'],
    bins=[20, 30, 40, 50, 60, 70, 100],  # Define the bins
    labels=['20-29', '30-39', '40-49', '50-59', '60-69', '70+'],  # Labels for the bins
    right=False  # Exclude the right edge
)




#count the number of loans per age group in our dataset
loan_counts = data['age_group'].value_counts().sort_index()

# Plot the bar chart
loan_counts.plot(kind='bar', color='skyblue', figsize=(10, 6))
plt.title('Number of Loans by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Loans')
plt.xticks(rotation=0)  # Keep x-axis labels horizontal
plt.show()



#preprocessing the data

# define the target and features
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# preprocessing pipeline
numerical_features = ['person_age', 'person_income', 'loan_amnt', 'loan_percent_income']
categorical_features = ['loan_intent', 'person_home_ownership', 'person_education']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


# train and evaluate our models

#starting with Logisticl regression


# Logistic Regression Pipeline
logreg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# train Logistic Regression
logreg_pipeline.fit(X_train, y_train)
y_pred_logreg = logreg_pipeline.predict(X_test)

# evaluate Logistic Regression
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))
print(f"ROC AUC Score: {roc_auc_score(y_test, logreg_pipeline.predict_proba(X_test)[:, 1])}")


