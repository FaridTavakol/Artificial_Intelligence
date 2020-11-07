"""
Problem 2, Logistic Regression with Regularization

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from Logistic_Regression import LogisticRegression

# Import Chronic Kidney Disease dataset
df = pd.read_csv(
    "chronic_kidney_disease_W_header_missing_category_replaced_and_given_value.csv")
# replacing the class values (ckd = 1, notckd = 0)
df = df.replace('ckd', 1)
df = df.replace('notckd', 0)

X = df.iloc[:, 0:24]  # features vectors
# class labels: ckd = Chronic Kidney Disease, notckd = Not Chronic Kidney Disease
y = df.iloc[:, 24]

# Replace missing feature values. 'median' is used for numerical cases and feature value
X = X.replace('?', np.nan)
X.to_csv('X_test_file.csv')
y.to_csv('Y_test_file.csv')
imr = SimpleImputer(missing_values=np.nan, strategy='median')
imr = imr.fit(X)
X_imputed = imr.transform(X.values)


le = LabelEncoder()  # positive class = 1 (ckd), negative class = 0 (notckd)
y = le.fit_transform(y)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=7)
# Z-score normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Principle component analysis (dimensionality reduction)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Training logistic regression classifier with L2 penalty
LR = LogisticRegression(
    learningRate=0.1, numEpochs=10, penalty='L2', C=0.01)  # range from 0.01 - 0.03
LR.train(X_train_pca, y_train, tol=10 ** -3)
# LR.plotCost()


# Testing fitted model on test data with cutoff probability 50%
predictions, probs = LR.predict(X_test_pca, 0.5)
performance = LR.performanceEval(predictions, y_test)
# LR.plotDecisionRegions(X_test_pca, y_test)
# LR.predictionPlot(X_test_pca, y_test)


# Print out performance values
for key, value in performance.items():
    print('%s : %.2f' % (key, value))
