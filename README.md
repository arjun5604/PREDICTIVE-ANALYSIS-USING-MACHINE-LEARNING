# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

**COMPANY**: CODETECH IT SOLUTIONS

**NAME**: ARJUN BHAGAVAN LAGAD 

**INTERN ID**: CT08HJY

**DOMAIN**: DATA ANALYTICS

**DURATION**: 4 WEEKS 

**MENTOR**: NEELA SANTHOSH

**DESCRIPTION OF TASK 2**: 
TASK 2 : Titanic Survival Prediction using a Random Forest Classifier
Objective:
The goal of this project is to build a machine learning model to predict the survival of passengers on the Titanic based on various features like age, sex, class, and other relevant information. The model will then be used to predict the survival of passengers in a test dataset.
Methodology:
 * Data Loading and Preprocessing:
   * The project starts by loading the training and test datasets from CSV files using pandas.
   * Data cleaning is performed to handle missing values and inconsistencies. This might involve:
     * Imputing missing values in the 'Age' column using techniques like mean imputation or more sophisticated methods.
     * Handling missing values in other columns like 'Embarked' based on their nature.
     * Converting categorical features like 'Sex' and 'Embarked' into numerical representations using one-hot encoding.
 * Feature Engineering:
   * New features might be engineered from existing ones to improve model performance. For example:
     * Creating a 'FamilySize' feature by combining 'SibSp' (siblings/spouses) and 'Parch' (parents/children).
     * Extracting 'Title' from the 'Name' column and using it as a feature.
 * Data Splitting:
   * The training data is split into training and validation sets using train_test_split from scikit-learn. This allows for evaluating the model's performance on unseen data before making predictions on the test set.
 * Model Training:
   * A Random Forest Classifier is initialized with appropriate hyperparameters (e.g., random_state for reproducibility).
   * The model is trained on the training data using the fit() method.
 * Model Evaluation:
   * The trained model is used to make predictions on the validation set.
   * Various evaluation metrics are calculated, such as accuracy, precision, recall, F1-score, and confusion matrix, to assess the model's performance.
 * Prediction on Test Data:
   * The model is used to predict the survival of passengers in the test dataset.
   * The predictions are saved to a CSV file in the required format (PassengerId, Survived).
Code Explanation:
 * Data Loading:
   train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

 * Data Preprocessing and Feature Engineering:
   # (This part is not shown in the provided code snippets)
# ... data cleaning and feature engineering steps ...

 * Data Splitting:
   X = train_data_cleaned.drop(columns=["Survived"])  # Features
y = train_data_cleaned["Survived"]                # Target variable
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

 * One-Hot Encoding:
   X_train = pd.get_dummies(X_train, columns=['Sex'])

 * Model Training:
   rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

 * Prediction and Evaluation:
   y_val_pred = rf_model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_val_pred))
# ... other evaluation metrics ...

 * Prediction on Test Data:
   test_predictions = rf_model.predict(test_data_cleaned)
test_output = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_predictions
})
test_output.to_csv("titanic_test_predictions.csv", index=False)

Error Handling:
The code encountered errors related to feature names not matching between the training and validation/test data due to the one-hot encoding of the 'Sex' feature. These errors were addressed by:
 * Ensuring consistent column order in the validation and test data by using the feature_names_in_ attribute of the trained model.
 * Performing one-hot encoding on the 'Sex' column in both the validation and test data.
Overall, this project demonstrates a basic workflow for building a machine learning model to predict the survival of passengers on
the Titanic using a Random Forest Classifier. By carefully preprocessing the data, engineering features, and handling model training and evaluation, the project aims to achieve a reasonable level of accuracy in predicting survival.
Further Improvements:
 * Hyperparameter tuning: Experiment with different hyperparameters for the Random Forest Classifier (e.g., number of trees, maximum depth) to potentially improve performance.
 * Feature importance analysis: Investigate which features are most important for predicting survival.
 * Ensemble methods: Explore other ensemble methods like Gradient Boosting or XGBoost to potentially achieve better results.
 * Advanced preprocessing techniques: Investigate more advanced preprocessing techniques such as standardization, normalization, or feature scaling.

**OUTPUT**: ![Image](https://github.com/user-attachments/assets/90ba5801-ff2b-4896-9298-aed35ef79c88)
