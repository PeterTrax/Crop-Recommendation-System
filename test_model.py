import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the dataset
file_path = 'Crop_recommendation.csv'  # Ensure this path is correct relative to your script location
df = pd.read_csv(file_path)

# Features and target variable
X = df.drop('label', axis=1)
y = df['label']

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(rf, X, y, cv=10)

# Calculate mean and standard deviation of the cross-validation scores
mean_cv_score = np.mean(cv_scores)
std_cv_score = np.std(cv_scores)

# Print the cross-validation results in a presentable format
print("Cross-validation scores:")
for fold_idx, score in enumerate(cv_scores, 1):
    print(f"  Fold {fold_idx}: {score:.4f}")

print(f"\nMean cross-validation score: {mean_cv_score:.4f}")
print(f"Standard deviation of cross-validation scores: {std_cv_score:.4f}")

# If you wish to export the trained model for later use, uncomment the lines below
# and run the code in order to create random_forest_model.pkl

#rf.fit(X, y)
#import joblib
#joblib.dump(rf, 'random_forest_model.pkl')
