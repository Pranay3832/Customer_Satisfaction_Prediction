# Customer Satisfaction Prediction

Predicts customer satisfaction using machine learning on support ticket data. Includes EDA, feature engineering, model training (RF, GB, XGB), hyperparameter tuning, and evaluation.

## Dataset

The project uses the `customer_support_tickets.csv` dataset, containing information about customer support interactions, product purchases, ticket details (like type, subject, channel, priority, status), resolution times, and customer satisfaction ratings.

## Setup

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd <repository-name>
    ```
3.  Install required libraries:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn xgboost
    ```
    *(Note: Ensure you have Python and pip installed.)*

## Usage

Open and run the `Customer_Satisfaction_Prediction (2).ipynb` notebook using Jupyter Notebook, JupyterLab, Google Colab, or a compatible environment. The notebook contains the complete workflow from data loading to model evaluation.

## Project Workflow

1.  **Import Libraries:** Import necessary Python libraries for data manipulation, preprocessing, modeling, and visualization.
2.  **Load Data:** Load the `customer_support_tickets.csv` file into a pandas DataFrame.
3.  **Exploratory Data Analysis (EDA) & Preprocessing:**
    * Initial data inspection (`.head()`, `.info()`, `.shape()`).
    * Handle missing data (analysis suggests satisfaction ratings might only be present for 'Closed' tickets).
    * Convert relevant columns to datetime objects (`First Response Time`, `Time to Resolution`).
4.  **Feature Engineering:**
    * Create `Time to Resolution (Hours)` feature.
    * Create basic NLP features from ticket subjects (`Ticket Subject Length`, `Ticket Subject Words`).
    * Define numerical and categorical features for preprocessing.
    * Set up a preprocessing pipeline using `ColumnTransformer` with `StandardScaler` for numerical features and `OneHotEncoder` for categorical features.
5.  **Model Training:**
    * Split the data into training and testing sets (implied by use of `X_train`, `X_test` etc.).
    * Define pipelines combining the preprocessor and various classification models (Random Forest, Gradient Boosting, XGBoost).
    * Train the models on the training data.
6.  **Hyperparameter Tuning:**
    * Use `GridSearchCV` to find the best hyperparameters for a chosen model (e.g., RandomForest or XGBoost demonstrated).
7.  **Evaluation:**
    * Evaluate model performance on the test set using metrics like accuracy and classification report.
    * Analyze feature importances from the best model.
8.  **Visualization:**
    * Plot distributions of the target variable (`Customer Satisfaction Rating`), `Customer Age`, and `Customer Gender`.
    * Analyze relationships between features (`Ticket Priority`, `Ticket Channel`) and the target variable using count plots.
    * Visualize the top 10 most purchased products.
    * Plot feature importances.

## Results

* The models were trained to predict the `Customer Satisfaction Rating` (treated as a classification problem with classes 0.0 to 4.0 in the evaluation shown).
* Evaluation metrics (Accuracy, Precision, Recall, F1-score) are presented in the notebook for the tested models (e.g., XGBoost, tuned model). Initial results showed accuracy around 18%, suggesting the prediction task is challenging with the current features/data.
* Feature importance analysis helps identify the factors most influential in predicting customer satisfaction according to the model.

## Author

* Pranay Kudale
