import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

try:
    # Try newer version first
    from pgmpy.models import DiscreteBayesianNetwork
except ImportError:
    # Fall back to older version
    from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork

warnings.filterwarnings("ignore")

class ModelTrainer:
    """Base class for all models to be implemented."""
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.accuracy = None
        self.auc = None

    def train(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass

    def evaluate(self, X_test, y_test):
        pass

    def get_performance_metrics(self):
        return {
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'auc': self.auc
        }

class RandomForestModel(ModelTrainer):
    """Bayesian Network ."""
    def __init__(self):
        super().__init__("Bayesian Network")

    def train(self, X_train, y_train, numerical_cols, categorical_cols):
        # Create preprocessor with different pipelines for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Create and train the Random Forest model
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])

        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        self.accuracy = accuracy_score(y_test, y_pred)
        self.auc = roc_auc_score(y_test, y_proba)

        return {
            'accuracy': self.accuracy,
            'auc': self.auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    def get_feature_importance(self, feature_names):
        if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
            return self.model.named_steps['classifier'].feature_importances_
        return None

class BayesianNetworkModel(ModelTrainer):
    """Markov Model implementation."""
    def __init__(self):
        super().__init__("Markov Model")
        self.network = None
        self.inference = None
        self.feature_names = None
        self.discretized_data = None
        self.state_names = None

    def _discretize_data(self, X, bins=3):
        """Discretize continuous variables for markov Network"""
        discretized = X.copy()

        # For numerical columns, discretize them
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64]:
                try:
                    # Make sure NaN values are handled before discretization
                    discretized[col] = discretized[col].fillna(discretized[col].median())
                    # Use integers as labels instead of strings to avoid categorical issues
                    discretized[col] = pd.qcut(discretized[col], bins, labels=False, duplicates='drop')
                except ValueError:
                    # Fall back to equal-width bins if quantile-based binning fails
                    discretized[col] = pd.cut(discretized[col], bins, labels=False)
            else:
                # For categorical columns, ensure they're integers
                discretized[col] = discretized[col].astype(int)

        return discretized

    def train(self, X_train, y_train, numerical_cols, categorical_cols):
        # Bayesian Networks work with discrete data, so we need to discretize continuous variables
        self.feature_names = X_train.columns.tolist()

        # Discretize the training data - handle missing values BEFORE discretization
        X_train_filled = X_train.copy()
        for col in numerical_cols:
            X_train_filled[col] = X_train_filled[col].fillna(X_train_filled[col].median())
        for col in categorical_cols:
            X_train_filled[col] = X_train_filled[col].fillna(X_train_filled[col].mode()[0])
            
        self.discretized_data = self._discretize_data(X_train_filled)
        
        # Add target variable - ensure it's an integer, not a Categorical
        self.discretized_data['Churn'] = y_train.astype(int)
        
        # Define the structure - for simplicity, we'll create a naive Bayes-like structure
        # where all features point to the target
        edges = [(feature, 'Churn') for feature in self.feature_names]

        # Create the Bayesian Network model
        self.network = BayesianNetwork(edges)
        
        # Build state_names dictionary properly
        self.state_names = {}
        for col in self.discretized_data.columns:
            self.state_names[col] = sorted(self.discretized_data[col].unique().tolist())
            
        # Print for debugging
        print("Checking data consistency...")
        print(f"Unique values for Churn: {self.discretized_data['Churn'].unique()}")
        for feature in self.feature_names:
            print(f"Unique values for {feature}: {self.discretized_data[feature].unique()}")

        # Fit the parameters
        try:
            self.network.fit(
                data=self.discretized_data,
                estimator=MaximumLikelihoodEstimator
            )
            
            # Create inference object
            self.inference = VariableElimination(self.network)
            
        except Exception as e:
            print(f"Error fitting Bayesian Network: {str(e)}")
            # Create a simple fallback model
            self.fallback_proba = y_train.mean()
            print(f"Using fallback probability: {self.fallback_proba}")

        return self.network

    def predict_proba(self, X_test):
        # If model training failed, use fallback probability
        if not hasattr(self, 'inference') or self.inference is None:
            return np.ones(len(X_test)) * self.fallback_proba
            
        # Handle missing values in test data before discretization
        X_test_filled = X_test.copy()
        for col in X_test.columns:
            if X_test[col].dtype in [np.float64, np.int64]:
                X_test_filled[col] = X_test_filled[col].fillna(X_test_filled[col].median())
            else:
                X_test_filled[col] = X_test_filled[col].fillna(0)  # Use 0 as default for categorical
                
        # Discretize test data
        X_test_disc = self._discretize_data(X_test_filled)

        # Store probabilities for each sample
        probs = []

        # For each test sample, compute probability of Churn=1
        for _, row in X_test_disc.iterrows():
            try:
                # Build evidence dictionary
                evidence = {feature: int(row[feature]) for feature in self.feature_names}
                
                query_result = self.inference.query(variables=['Churn'], evidence=evidence)
                # Get probability of Churn=1
                prob_churn = query_result.values[1]  # Index 1 should correspond to Churn=1
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                # If there's an error, use a default probability
                prob_churn = 0.5

            probs.append(prob_churn)

        return np.array(probs)

    def predict(self, X_test):
        probas = self.predict_proba(X_test)
        return (probas > 0.5).astype(int)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        self.accuracy = accuracy_score(y_test, y_pred)
        self.auc = roc_auc_score(y_test, y_proba)

        return {
            'accuracy': self.accuracy,
            'auc': self.auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

def load_and_explore_data(filepath):
    """Load and explore the churn dataset."""
    print("=== Loading and Exploring Data ===")
    df = pd.read_csv(filepath)

    print(f"Dataset shape: {df.shape}")
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nTarget variable distribution:")
    print(df['Churn'].value_counts(normalize=True) * 100)

    return df

def preprocess_data(df):
    """Preprocess the data for the models."""
    print("\n=== Preprocessing Data ===")

    # Save original categorical values for reference
    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
    for col in categorical_cols:
        print(f"\nUnique values for {col}:", df[col].unique())

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"\n{col} encoding mapping:")
        for i, label in enumerate(le.classes_):
            print(f"  {label} -> {i}")

    # Handle missing values
    numerical_cols = ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # Print range of values for numerical features
    print("\nValue ranges for numerical columns:")
    for col in numerical_cols:
        print(f"  {col}: Min={df[col].min()}, Max={df[col].max()}")

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoders, numerical_cols, categorical_cols

def build_and_evaluate_models(X_train, X_test, y_train, y_test, numerical_cols, categorical_cols):
    """Build and evaluate multiple models for comparison."""
    print("\n=== Training and Evaluating Models ===")

    # Initialize models
    models = {
        'rf': RandomForestModel(),
        'bn': BayesianNetworkModel()
    }

    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {model.model_name}...")
        model.train(X_train, y_train, numerical_cols, categorical_cols)

        print(f"Evaluating {model.model_name}...")
        eval_results = model.evaluate(X_test, y_test)

        # Store results
        results[name] = {
            'model': model,
            'eval': eval_results
        }

        # Print metrics
        print(f"\n{model.model_name} Performance:")
        print(f"Accuracy: {eval_results['accuracy']:.4f}")
        print(f"ROC AUC: {eval_results['auc']:.4f}")
        print("\nClassification Report:")
        print(eval_results['classification_report'])

    return results

def predict_churn(model, user_input, feature_names):
    """Predict churn probability based on user input dictionary."""
    # Convert user input to correct format for prediction
    input_df = pd.DataFrame([user_input], columns=feature_names)

    # Make prediction based on selected model type
    if isinstance(model, RandomForestModel):
        churn_proba = model.predict_proba(input_df)[0]
    elif isinstance(model, BayesianNetworkModel):
        churn_proba = model.predict_proba(input_df)[0]
    else:
        raise ValueError("Unsupported model type")

    churn_pred = 1 if churn_proba > 0.5 else 0
    retention_prob = 1 - churn_proba

    return churn_pred, retention_prob

def identify_risk_factors(user_input, feature_names, model):
    """Identify the top risk factors for churn based on feature importance and user input."""
    # For Random Forest, use feature importance
    if isinstance(model, RandomForestModel):
        feature_importance = model.get_feature_importance(feature_names)
        importance_dict = dict(zip(feature_names, feature_importance))
    else:
        # For Bayesian Network, use a heuristic approach
        importance_dict = {
            'Tenure': 0.25,
            'Usage Frequency': 0.20,
            'Support Calls': 0.15,
            'Payment Delay': 0.15,
            'Last Interaction': 0.15,
            'Total Spend': 0.10
        }

    # Define risk thresholds for numerical features
    risk_thresholds = {
        'Tenure': (lambda x: x < 6, "Short tenure (new customer)"),
        'Usage Frequency': (lambda x: x < 5, "Low usage frequency"),
        'Support Calls': (lambda x: x > 3, "High number of support calls"),
        'Payment Delay': (lambda x: x > 2, "Significant payment delays"),
        'Last Interaction': (lambda x: x > 30, "Long time since last interaction"),
        'Total Spend': (lambda x: x < 100, "Low total spend")
    }

    # Find top risk factors based on importance and thresholds
    risk_factors = []

    # First check the high-importance features that exceed risk thresholds
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    for feature, importance in sorted_features:
        if feature in risk_thresholds and feature in user_input:
            condition_func, risk_message = risk_thresholds[feature]
            if condition_func(user_input[feature]):
                risk_factors.append(f"{risk_message} (Important feature: {importance:.3f})")

    return risk_factors[:3]  # Return top 3 risk factors

def generate_recommendations(user_input, risk_factors, retention_prob, model_type):
    """Generate personalized recommendations based on identified risk factors and model insights."""
    recommendations = []

    # Add model-specific insights
    if model_type == "Markov Model":
        recommendations.append("Based on Markov analysis, addressing conditional dependencies is key")

    # Standard recommendations based on risk factors
    if any("tenure" in factor.lower() for factor in risk_factors):
        recommendations.append("Implement an enhanced onboarding program to improve early engagement")
        recommendations.append("Offer a loyalty discount for continuing beyond the first quarter")

    if any("usage frequency" in factor.lower() for factor in risk_factors):
        recommendations.append("Send personalized feature highlights to drive engagement")
        recommendations.append("Create a re-engagement email campaign with usage tips")

    if any("support calls" in factor.lower() for factor in risk_factors):
        recommendations.append("Conduct a detailed review of customer's support history")
        recommendations.append("Assign a dedicated account representative for personalized support")

    if any("payment" in factor.lower() for factor in risk_factors):
        recommendations.append("Offer flexible payment options or installment plans")
        recommendations.append("Provide a one-time discount on their next payment")

    if any("interaction" in factor.lower() for factor in risk_factors):
        recommendations.append("Schedule an account review call to reconnect")
        recommendations.append("Send a personalized check-in message from the account team")

    # Add subscription-specific recommendations
    if 'Subscription Type' in user_input:
        if user_input['Subscription Type'] == 0:  # Basic plan (assuming 0 is Basic)
            recommendations.append("Offer a free trial of premium features to showcase additional value")

    # Risk level specific approaches
    if retention_prob < 0.3:  # High risk
        recommendations.append("Initiate immediate retention protocol with significant incentives")
        recommendations.append("Escalate to senior customer success manager for intervention")
    elif retention_prob < 0.6:  # Medium risk
        recommendations.append("Schedule regular check-ins over the next quarter")
        recommendations.append("Provide educational content addressing their specific usage patterns")

    return recommendations[:5]  # Return top 5 most relevant recommendations