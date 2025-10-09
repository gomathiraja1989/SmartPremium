import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
import warnings
warnings.filterwarnings('ignore')

# Set backend for matplotlib to avoid Tkinter issues
import matplotlib
matplotlib.use('Agg')


class InsurancePremiumPredictor:
    def __init__(self, random_state=42, experiment_name="SmartPremium_Insurance"):
        self.random_state = random_state
        self.scaler = None
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        
        # Set up MLflow
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
        # Set visualization style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def load_data(self, train_path, test_path, sample_size=0.2):
        print("=== STEP 1: LOADING DATA ===")
        
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Original data - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Sample only training data for faster processing
        if sample_size < 1.0:
            train_df = train_df.sample(frac=sample_size, random_state=self.random_state)
            print(f"Sampled training data: {train_df.shape}")
            print(f"Full test data maintained: {test_df.shape}")
        
        return train_df, test_df
    
    def preprocess_data(self, train_df, test_df):
        print("\n=== STEP 2: DATA PREPROCESSING ===")
        
        # Create copies to avoid modifying original data
        train_clean = train_df.copy()
        test_clean = test_df.copy()
        
        # Drop columns with excessive missing values
        columns_to_drop = [
            'id', 'Occupation', 'Previous Claims', 'Customer Feedback', 
            'Policy Start Date'
        ]
        train_clean = train_clean.drop(columns=columns_to_drop, errors='ignore')
        test_clean = test_clean.drop(columns=columns_to_drop, errors='ignore')
        
        print("Dropped columns with excessive missing values")
        
        # Handle missing values
        self._handle_missing_values(train_clean, test_clean)
        
        return train_clean, test_clean
    
    def _handle_missing_values(self, train_df, test_df):
        # Handle numerical columns
        numerical_columns = train_df.select_dtypes(include=[np.number]).columns
        
        for column in numerical_columns:
            if train_df[column].isnull().any():
                median_value = train_df[column].median()
                train_df[column].fillna(median_value, inplace=True)
                if column in test_df.columns:
                    test_df[column].fillna(median_value, inplace=True)
                print(f"Filled missing values in {column} with median: {median_value:.2f}")
        
        # Handle categorical columns
        categorical_columns = train_df.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            if train_df[column].isnull().any():
                mode_value = train_df[column].mode()[0]
                train_df[column].fillna(mode_value, inplace=True)
                if column in test_df.columns:
                    test_df[column].fillna(mode_value, inplace=True)
                print(f"Filled missing values in {column} with mode: {mode_value}")
    
    def perform_eda(self, train_df, target_column='Premium Amount'):
        print("\n=== STEP 3: EXPLORATORY DATA ANALYSIS ===")
        
        # 1. Target variable distribution
        self._plot_target_distribution(train_df, target_column)
        
        # 2. Numerical features distribution
        self._plot_numerical_distributions(train_df)
        
        # 3. Categorical features distribution
        self._plot_categorical_distributions(train_df)
        
        # 4. Correlation matrix
        self._plot_correlation_matrix(train_df, target_column)
        
        print("Exploratory Data Analysis completed")
    
    def _plot_target_distribution(self, df, target_column):
        """Plot distribution of the target variable."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original distribution
        ax1.hist(df[target_column], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Original Premium Distribution', fontweight='bold')
        ax1.set_xlabel('Premium Amount ($)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Log-transformed distribution
        ax2.hist(np.log1p(df[target_column]), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('Log-Transformed Premium Distribution', fontweight='bold')
        ax2.set_xlabel('Log(Premium Amount + 1)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Square root transformed distribution
        ax3.hist(np.sqrt(df[target_column]), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Square Root Premium Distribution', fontweight='bold')
        ax3.set_xlabel('Sqrt(Premium Amount)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_numerical_distributions(self, df):
        """Plot distributions of key numerical features."""
        numerical_features = ['Age', 'Annual Income', 'Health Score', 'Credit Score']
        numerical_features = [f for f in numerical_features if f in df.columns]
        
        if not numerical_features:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(numerical_features):
            axes[i].hist(df[feature], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
            axes[i].set_title(f'Distribution of {feature}', fontweight='bold')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(numerical_features), 4):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('outputs/numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_categorical_distributions(self, df):
        """Plot distributions of key categorical features."""
        categorical_features = ['Gender', 'Policy Type', 'Location', 'Smoking Status']
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        if not categorical_features:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(categorical_features):
            value_counts = df[feature].value_counts()
            axes[i].bar(value_counts.index, value_counts.values, color='lightcoral', edgecolor='black')
            axes[i].set_title(f'Distribution of {feature}', fontweight='bold')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
        
        # Remove empty subplots
        for i in range(len(categorical_features), 4):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('outputs/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_correlation_matrix(self, df, target_column):
        """Plot correlation matrix for numerical features."""
        numerical_df = df.select_dtypes(include=[np.number])
        
        if len(numerical_df.columns) < 2:
            return
        
        # Limit to top 10 features to avoid overcrowding
        numerical_df = numerical_df.iloc[:, :10]
        
        plt.figure(figsize=(12, 8))
        correlation_matrix = numerical_df.corr()
        
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            fmt='.2f', 
            linewidths=0.5,
            cbar_kws={"shrink": .8}
        )
        plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Show correlations with target variable
        if target_column in correlation_matrix.columns:
            target_correlations = correlation_matrix[target_column].sort_values(ascending=False)
            print("\nTop correlations with Premium Amount:")
            print(target_correlations.head(10))
    
    def engineer_features(self, train_df, test_df):
        """
        Perform feature engineering on training and test data.
        """
        print("\n=== STEP 4: FEATURE ENGINEERING ===")
        
        train_engineered = train_df.copy()
        test_engineered = test_df.copy()
        
        # Create interaction features
        self._create_interaction_features(train_engineered, test_engineered)
        
        # Create demographic features
        self._create_demographic_features(train_engineered, test_engineered)
        
        # Encode categorical variables
        self._encode_categorical_features(train_engineered, test_engineered)
        
        print("Feature engineering completed")
        
        return train_engineered, test_engineered
    
    def _create_interaction_features(self, train_df, test_df):
        """Create interaction features between important variables."""
        # Age-Health interaction
        if all(col in train_df.columns for col in ['Age', 'Health Score']):
            train_df['Age_Health_Interaction'] = train_df['Age'] * train_df['Health Score']
            test_df['Age_Health_Interaction'] = test_df['Age'] * test_df['Health Score']
        
        # Income-Credit interaction
        if all(col in train_df.columns for col in ['Annual Income', 'Credit Score']):
            train_df['Income_Credit_Interaction'] = (
                train_df['Annual Income'] * train_df['Credit Score'] / 1000
            )
            test_df['Income_Credit_Interaction'] = (
                test_df['Annual Income'] * test_df['Credit Score'] / 1000
            )
    
    def _create_demographic_features(self, train_df, test_df):
        """Create demographic and risk-based features."""
        # Age groups
        if 'Age' in train_df.columns:
            age_bins = [0, 25, 35, 45, 55, 65, 100]
            age_labels = [1, 2, 3, 4, 5, 6]
            train_df['Age_Group'] = pd.cut(train_df['Age'], bins=age_bins, labels=age_labels).astype(int)
            test_df['Age_Group'] = pd.cut(test_df['Age'], bins=age_bins, labels=age_labels).astype(int)
        
        # Income categories
        if 'Annual Income' in train_df.columns:
            income_bins = [0, 30000, 60000, 90000, 120000, float('inf')]
            income_labels = [1, 2, 3, 4, 5]
            train_df['Income_Category'] = pd.cut(
                train_df['Annual Income'], bins=income_bins, labels=income_labels
            ).astype(int)
            test_df['Income_Category'] = pd.cut(
                test_df['Annual Income'], bins=income_bins, labels=income_labels
            ).astype(int)
        
        # Risk flags
        if 'Age' in train_df.columns:
            train_df['Is_Young_Driver'] = (train_df['Age'] < 25).astype(int)
            train_df['Is_Senior'] = (train_df['Age'] > 60).astype(int)
            test_df['Is_Young_Driver'] = (test_df['Age'] < 25).astype(int)
            test_df['Is_Senior'] = (test_df['Age'] > 60).astype(int)
    
    def _encode_categorical_features(self, train_df, test_df):
        """Encode categorical features using Label Encoding."""
        categorical_columns = train_df.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            le = LabelEncoder()
            
            # Combine train and test for consistent encoding
            combined_data = pd.concat([train_df[column], test_df[column]], axis=0)
            le.fit(combined_data.astype(str))
            
            # Transform both datasets
            train_df[column] = le.transform(train_df[column].astype(str))
            test_df[column] = le.transform(test_df[column].astype(str))
            
            self.label_encoders[column] = le
            
            print(f"Encoded categorical feature: {column}")
    
    def prepare_features(self, train_df, test_df, target_column='Premium Amount'):
        print("\n=== STEP 5: FEATURE PREPARATION ===")
        
        # Select only numerical features
        self.feature_columns = [
            col for col in train_df.columns 
            if col != target_column 
            and train_df[col].dtype in ['int64', 'float64']
            and col in test_df.columns
        ]
        
        X_train = train_df[self.feature_columns]
        y_train = train_df[target_column]
        X_test = test_df[self.feature_columns]
        
        print(f"Selected {len(self.feature_columns)} features for modeling")
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train
    
    def train_model(self, X_train, y_train):
        print("\n=== STEP 6: MODEL TRAINING WITH MLFLOW ===")
        
        # Split data for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )
        
        # Initialize models with parameters for MLflow tracking
        models = {
            'Random Forest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 15,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10
                }
            },
            'XGBoost': {
                'model': XGBRegressor(
                    n_estimators=100,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    verbosity=0
                ),
                'params': {
                    'n_estimators': 100,
                    'max_depth': 8,
                    'learning_rate': 0.1
                }
            }
        }
        
        best_model = None
        best_r2 = -float('inf')
        best_model_name = ""
        results = {}
        
        # Train and evaluate each model in separate MLflow runs
        for name, model_info in models.items():
            print(f"\nTraining {name}...")
            
            # Start a new run for each model
            with mlflow.start_run(run_name=f"{name}_Training", nested=True):
                model = model_info['model']
                params = model_info['params']
                
                # Log model parameters
                mlflow.log_params(params)
                mlflow.log_param("model_type", name)
                mlflow.log_param("random_state", self.random_state)
                mlflow.log_param("feature_count", len(self.feature_columns))
                
                # Train model
                model.fit(X_tr, y_tr)
                
                # Validate model
                y_pred = model.predict(X_val)
                
                # Calculate performance metrics
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                results[name] = {
                    'model': model,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                # Log metrics to MLflow
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                
                # Prepare MLflow model signature and input example
                signature = infer_signature(X_val, y_pred)
                input_example = X_val[:5]

                # Log model with signature and example
                if name == "Random Forest":
                    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)
                else:
                    mlflow.xgboost.log_model(model, "model", signature=signature, input_example=input_example)
                
                print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
                
                # Update best model
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    best_model_name = name
        
        self.model = best_model
        print(f"\n🎯 Best Model: {best_model_name} with R²: {best_r2:.4f}")
        
        # Log best model summary in a separate run
        with mlflow.start_run(run_name="Best_Model_Summary", nested=True):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_r2", best_r2)
            mlflow.log_metric("best_rmse", results[best_model_name]['rmse'])
            mlflow.log_metric("best_mae", results[best_model_name]['mae'])
            mlflow.log_param("feature_count", len(self.feature_columns))
            
            # Log the best model
            # Reuse validation set for signature
            y_pred_best = best_model.predict(X_val)
            signature_best = infer_signature(X_val, y_pred_best)
            input_example_best = X_val[:5]

            if best_model_name == "Random Forest":
                mlflow.sklearn.log_model(best_model, "best_model", signature=signature_best, input_example=input_example_best)
            else:
                mlflow.xgboost.log_model(best_model, "best_model", signature=signature_best, input_example=input_example_best)
        
        return results[best_model_name]['rmse'], results[best_model_name]['mae'], best_r2
    
    def analyze_feature_importance(self):
        """Analyze and plot feature importance."""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            print("Model not trained or feature importance not available")
            return
        
        print("\n=== STEP 7: FEATURE IMPORTANCE ANALYSIS ===")
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        print(importance_df.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        
        plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
        plt.title('Top 15 Feature Importances', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Log feature importance to MLflow
        with mlflow.start_run(run_name="Feature_Importance_Analysis", nested=True):
            mlflow.log_artifact("outputs/feature_importance.png")
            # Log top features
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                mlflow.log_param(f"top_feature_{i+1}", row['feature'])
                mlflow.log_metric(f"importance_{i+1}", row['importance'])
        
        return importance_df
    
    def make_predictions(self, X_test, sample_submission_path):
        print("\n=== STEP 8: MAKING PREDICTIONS ===")
        
        # Make predictions on FULL test data
        print("Generating predictions for full test dataset...")
        predictions = self.model.predict(X_test)
        
        # Ensure predictions are in reasonable range
        predictions = np.clip(predictions, 500, 2500)
        
        print(f"Prediction range: ${predictions.min():.2f} to ${predictions.max():.2f}")
        print(f"Number of predictions: {len(predictions)}")
        
        # Create file
        sample_submission = pd.read_csv(sample_submission_path)
        
        print(f"File expected rows: {len(sample_submission)}")
        print(f"Predictions generated: {len(predictions)}")
        
        # Ensure we have the right number of predictions
        if len(predictions) != len(sample_submission):
            print(f"⚠️ Warning: Prediction count mismatch")
            print(f"   Expected: {len(sample_submission)}")
            print(f"   Got: {len(predictions)}")
            
            # If we have fewer predictions, we need to retrain on full data
            if len(predictions) < len(sample_submission):
                print("❌ Not enough predictions. Need to retrain on full training data.")
                return None
        
        # Assign predictions to file
        if 'Premium Amount' in sample_submission.columns:
            sample_submission['Premium Amount'] = predictions
        else:
            # Use the second column as target (assuming first is ID)
            sample_submission.iloc[:, 1] = predictions
        
        # Save file
        sample_submission.to_csv('outputs/final_submission.csv', index=False)
        print("✅ Submission file saved as 'final_submission.csv'")
        
        # Log info to MLflow
        with mlflow.start_run(run_name="Submission_Results", nested=True):
            mlflow.log_artifact("outputs/final_submission.csv")
            mlflow.log_metric("prediction_min", predictions.min())
            mlflow.log_metric("prediction_max", predictions.max())
            mlflow.log_metric("prediction_mean", predictions.mean())
            mlflow.log_metric("prediction_std", predictions.std())
            mlflow.log_param("submission_rows", len(sample_submission))
        
        return sample_submission
    
    def save_artifacts(self):
        """Save model and preprocessing artifacts."""
        import joblib
        
        print("\n=== STEP 9: SAVING ARTIFACTS ===")
        
        artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'random_state': self.random_state
        }
        
        joblib.dump(artifacts, 'outputs/model_artifacts.pkl')
        print("Model artifacts saved as 'model_artifacts.pkl'")
        
        # Log artifacts to MLflow
        with mlflow.start_run(run_name="Model_Artifacts", nested=True):
            mlflow.log_artifact("outputs/model_artifacts.pkl")
            mlflow.log_artifact("outputs/target_distribution.png")
            mlflow.log_artifact("outputs/correlation_matrix.png")
            mlflow.log_artifact("outputs/feature_importance.png")
    
    def run_complete_pipeline(self, train_path, test_path, sample_submission_path, sample_size=0.2):
        print("🚀 STARTING SMART PREMIUM PREDICTION PIPELINE")
        print("=" * 50)
        print("📊 MLflow Experiment:", self.experiment_name)
        print("=" * 50)
        
        # Start main MLflow run for the entire pipeline
        with mlflow.start_run(run_name="Complete_Pipeline"):
            mlflow.log_param("sample_size", sample_size)
            mlflow.log_param("train_path", train_path)
            mlflow.log_param("test_path", test_path)
            mlflow.log_param("pipeline_version", "1.0")
            
            # Step 1: Load data (sample only training data)
            train_df, test_df = self.load_data(train_path, test_path, sample_size)
            mlflow.log_param("train_samples", len(train_df))
            mlflow.log_param("test_samples", len(test_df))
            
            # Step 2: Preprocess data
            train_clean, test_clean = self.preprocess_data(train_df, test_df)
            
            # Step 3: Perform EDA
            self.perform_eda(train_clean)
            
            # Step 4: Engineer features
            train_engineered, test_engineered = self.engineer_features(train_clean, test_clean)
            mlflow.log_param("engineered_features", len(train_engineered.columns))
            
            # Step 5: Prepare features
            X_train, X_test, y_train = self.prepare_features(train_engineered, test_engineered)
            mlflow.log_param("final_feature_count", len(self.feature_columns))
            
            # Step 6: Train model (uses nested runs)
            rmse, mae, r2 = self.train_model(X_train, y_train)
            
            # Step 7: Analyze feature importance
            importance_df = self.analyze_feature_importance()
            
            # Step 8: Make predictions on FULL test data
            submission_df = self.make_predictions(X_test, sample_submission_path)
            
            if submission_df is None:
                print("❌ Failed to create file due to prediction count mismatch")
                # Log failure
                mlflow.log_param("submission_status", "failed")
                return rmse, r2, None, importance_df
            
            # Step 9: Save artifacts
            self.save_artifacts()
            
            # Log final metrics to main run
            mlflow.log_metric("final_rmse", rmse)
            mlflow.log_metric("final_mae", mae)
            mlflow.log_metric("final_r2", r2)
            mlflow.log_param("best_model_type", type(self.model).__name__)
            mlflow.log_param("submission_status", "success")
            mlflow.log_metric("prediction_count", len(submission_df))
        
        # Final summary
        print("\n" + "=" * 50)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("📊 FINAL RESULTS SUMMARY:")
        print(f"   • Best Model: {type(self.model).__name__}")
        print(f"   • Validation RMSE: {rmse:.2f}")
        print(f"   • Validation MAE: {mae:.2f}")
        print(f"   • Validation R²: {r2:.4f}")
        print(f"   • Features used: {len(self.feature_columns)}")
        print(f"   • Predictions range: ${submission_df['Premium Amount'].min():.2f} "
              f"to ${submission_df['Premium Amount'].max():.2f}")
        print(f"   • Submission file: 'final_submission.csv'")
        print(f"   • Artifacts saved: 'model_artifacts.pkl'")
        print(f"   • Visualizations saved: PNG files")
        print(f"   • MLflow Experiment: {self.experiment_name}")
        print(f"   • View MLflow UI: mlflow ui")
        
        return rmse, r2, submission_df, importance_df


def main():
    # Ensure no active runs
    if mlflow.active_run():
        mlflow.end_run()
    
    # Initialize the predictor
    predictor = InsurancePremiumPredictor(
        random_state=42,
        experiment_name="SmartPremium_Insurance_v1"
    )
    
    # File paths
    train_path = 'train.csv'
    test_path = 'test.csv'
    sample_submission_path = 'sample_submission.csv'
    
    try:
        # Run complete pipeline with MLflow integration
        rmse, r2, submission_df, importance_df = predictor.run_complete_pipeline(
            train_path=train_path,
            test_path=test_path,
            sample_submission_path=sample_submission_path,
            sample_size=0.1  # Use only 10% of training data for speed
        )
        
        if submission_df is not None:
            print("📁 Files created:")
            print("   - final_submission.csv")
            print("   - model_artifacts.pkl (Trained model and preprocessors)")
            print("   - Multiple visualization PNG files")
        else:
            print("\n❌ file creation failed.")
            print("💡 Try increasing sample_size or using full training data.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        print("Please ensure all required CSV files are in the current directory:")
        print("  - train.csv")
        print("  - test.csv") 
        print("  - sample_submission.csv")
    
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        
        # Ensure MLflow run is ended on error
        if mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    main()