import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')



class DataProcessorBuilder:
    """Builder pattern for modular data processing and feature engineering"""
    
    def __init__(self):
        """Initialize with empty pipeline configuration"""
        self.pipeline_steps = []
        self.feature_columns = []
        self._time_features_config = None
        self._lag_features_config = None
        self._rolling_features_config = None
        self._event_features_config = None
        self._country_features_enabled = False
        self._season_features_enabled = False
        print(f"🏗️  DataProcessorBuilder initialized with empty pipeline")
    
    def add_time_features(self):
        """Configure time-based features to be added"""
        print("   ⏰ Configuring time features...")
        self._time_features_config = True
        self.pipeline_steps.append('time_features')
        
        # Pre-define the feature columns that will be created
        time_feature_cols = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter', 
                            'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 
                            'dayofyear_sin', 'dayofyear_cos', 'is_weekend']
        self.feature_columns.extend(time_feature_cols)
        return self
    
    def add_lag_features(self, lags=[1, 2, 3, 7, 14, 28]):
        """Configure lag features for revenue by store"""
        print(f"   📈 Configuring lag features: {lags}")
        self._lag_features_config = lags
        self.pipeline_steps.append('lag_features')
        
        # Pre-define the feature columns that will be created
        lag_feature_cols = [f'revenue_lag_{lag}' for lag in lags]
        self.feature_columns.extend(lag_feature_cols)
        return self
    
    def add_rolling_features(self, windows=[3, 7, 14, 28]):
        """Configure rolling window statistics"""
        print(f"   📊 Configuring rolling features: {windows}")
        self._rolling_features_config = windows
        self.pipeline_steps.append('rolling_features')
        
        # Pre-define the feature columns that will be created
        rolling_feature_cols = []
        for window in windows:
            rolling_feature_cols.extend([
                f'revenue_rolling_mean_{window}',
                f'revenue_rolling_std_{window}'
            ])
        self.feature_columns.extend(rolling_feature_cols)
        return self
    
    def add_event_features(self, significant_events=None):
        """Configure event-based features"""
        if significant_events is None:
            raise ValueError("significant_events must be provided")

        print(f"   🎉 Configuring event features for {len(significant_events)} significant events")
        self._event_features_config = significant_events
        self.pipeline_steps.append('event_features')
        
        # Pre-define the feature columns that will be created
        event_feature_cols = []
        for event_name in significant_events:
            clean_name = event_name.replace(",", "").replace(" ", "_").replace("'", "")
            event_feature_cols.append(f'is_{clean_name}')
        event_feature_cols.append('has_other_event')
        self.feature_columns.extend(event_feature_cols)
        return self
    
    def add_country_features(self):
        """Configure country-based one-hot encoded features"""
        print("   🌍 Configuring country features...")
        self._country_features_enabled = True
        self.pipeline_steps.append('country_features')
        # Note: Country feature columns will be determined dynamically based on data
        return self
    
    def add_season_features(self):
        """Configure season-based features"""
        print("   🍂 Configuring season features...")
        self._season_features_enabled = True
        self.pipeline_steps.append('season_features')
        
        # Pre-define the feature columns that will be created
        season_feature_cols = ['season_Winter', 'season_Spring', 'season_Summer', 'season_Fall']
        self.feature_columns.extend(season_feature_cols)
        return self
    
    def process(self, df):
        """Apply the configured pipeline to the given dataframe"""
        print(f"🔧 Processing dataframe with {len(df)} rows using configured pipeline...")
        
        # Work with a copy to avoid modifying the original
        processed_df = df.copy()
        processed_df = processed_df.sort_values(['store_id', 'date'])
        
        # Track actual feature columns created (may differ due to dynamic features like countries)
        actual_feature_columns = []
        
        for step in self.pipeline_steps:
            if step == 'time_features':
                processed_df, step_features = self._apply_time_features(processed_df)
                actual_feature_columns.extend(step_features)
                
            elif step == 'lag_features':
                processed_df, step_features = self._apply_lag_features(processed_df)
                actual_feature_columns.extend(step_features)
                
            elif step == 'rolling_features':
                processed_df, step_features = self._apply_rolling_features(processed_df)
                actual_feature_columns.extend(step_features)
                
            elif step == 'event_features':
                processed_df, step_features = self._apply_event_features(processed_df)
                actual_feature_columns.extend(step_features)
                
            elif step == 'country_features':
                processed_df, step_features = self._apply_country_features(processed_df)
                actual_feature_columns.extend(step_features)
                
            elif step == 'season_features':
                processed_df, step_features = self._apply_season_features(processed_df)
                actual_feature_columns.extend(step_features)
        
        print("   ✅ Data processing complete!")
        return processed_df, actual_feature_columns
    
    def _apply_time_features(self, df):
        """Apply time-based features"""
        # Basic time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['week'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # Cyclical encoding for better representation
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
        
        # Weekend flag
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter', 
                   'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 
                   'dayofyear_sin', 'dayofyear_cos', 'is_weekend']
        return df, features
    
    def _apply_lag_features(self, df):
        """Apply lag features"""
        features = []
        for lag in self._lag_features_config:
            df[f'revenue_lag_{lag}'] = df.groupby('store_id')['revenue'].shift(lag)
            features.append(f'revenue_lag_{lag}')
        return df, features
    
    def _apply_rolling_features(self, df):
        """Apply rolling window statistics"""
        features = []
        for window in self._rolling_features_config:
            # Rolling mean
            df[f'revenue_rolling_mean_{window}'] = (
                df.groupby('store_id')['revenue']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            features.append(f'revenue_rolling_mean_{window}')
            
            # Rolling standard deviation
            df[f'revenue_rolling_std_{window}'] = (
                df.groupby('store_id')['revenue']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
            features.append(f'revenue_rolling_std_{window}')
        return df, features
    
    def _apply_event_features(self, df):
        """Apply event-based features"""
        features = []
        
        # Significant event indicators
        for event_name in self._event_features_config:
            # Clean event name for column naming
            clean_name = event_name.replace(",", "").replace(" ", "_").replace("'", "")
            col_name = f'is_{clean_name}'
            df[col_name] = (df['event'] == event_name).astype(int)
            features.append(col_name)

        # General event indicator for other events
        df['has_other_event'] = (
            (df['event'].notna()) & 
            (~df['event'].isin(self._event_features_config))
        ).astype(int)
        features.append('has_other_event')
        return df, features
    
    def _apply_country_features(self, df):
        """Apply country-based one-hot encoded features"""
        features = []
        unique_countries = [c for c in df['country'].unique() if c != 'All Stores']
        for country in unique_countries:
            col_name = f'country_{country}'
            df[col_name] = (df['country'] == country).astype(int)
            features.append(col_name)
        return df, features
    
    def _apply_season_features(self, df):
        """Apply season-based features"""
        # Create season column first
        df['season'] = ((df['month'] % 12 + 3) // 3).map({
            1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'
        })
        
        # One-hot encode seasons (these are the actual features we'll use)
        features = []
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            col_name = f'season_{season}'
            df[col_name] = (df['season'] == season).astype(int)
            features.append(col_name)
        
        return df, features
    
    def get_feature_columns(self):
        """Get the list of feature columns (may not include dynamic features like countries)"""
        return self.feature_columns

class XGBoostModel:
    def __init__(self, data, data_processor: DataProcessorBuilder):
        self.data = data
        self.data_processor = data_processor  # Accept configured processor as dependency
        self.model = None
        self.feature_importance = None
        self.training_results = None
        self.feature_columns = None
        self.processed_data = None
        print(f"🏗️  XGBoostModel initialized with provided DataProcessorBuilder")

    def train_gradient_boosting_model(self, data):
        """Train Gradient Boosting model using entire dataset with DataProcessorBuilder"""
        
        print("🏗️  Creating features using provided DataProcessorBuilder for training...")
        
        # Use the provided data processor
        processed_data, feature_columns = self.data_processor.process(data)
        
        # Remove rows with NaN values (due to lag features)
        clean_data = processed_data.dropna().copy()
        print(f"   📊 Clean data: {len(clean_data)} rows (after removing NaN)")
        
        # Filter to existing columns only
        existing_features = [col for col in feature_columns if col in clean_data.columns]
        print(f"   🔧 Using {len(existing_features)} features")
        
        print(f"   📅 Training period: {clean_data['date'].min()} to {clean_data['date'].max()}")
        print(f"   📈 Total training samples: {len(clean_data)}")
        
        # Prepare features and target
        X_train = clean_data[existing_features]
        y_train = clean_data['revenue']
        
        # Handle any remaining NaN values
        X_train = X_train.fillna(0)
        
        # Train model
        print(f"   🔄 Training Gradient Boosting Model...")
        
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions on training data
        y_pred_train = model.predict(X_train)
        
        # Calculate training metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)
        
        print(f"\n📊 Model Performance (Training Data):")
        print(f"   🎯 Train MAE: ${train_mae:,.0f}")
        print(f"   📏 Train RMSE: ${train_rmse:,.0f}")
        print(f"   📈 Train R²: {train_r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': existing_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 15 Most Important Features:")
        print("-" * 50)
        for idx, row in feature_importance.head(15).iterrows():
            print(f"   {row['feature']:25s}: {row['importance']:.3f}")
        
        # Store results as instance variables for use by other methods
        self.model = model
        self.feature_importance = feature_importance
        self.training_results = (X_train, y_train, y_pred_train)
        self.feature_columns = existing_features
        self.processed_data = processed_data
        return self

    def plot_model_results(self):
        """Plot model performance and feature importance"""
        X_train, y_train, y_pred_train = self.training_results
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Feature Importance
        top_features = self.feature_importance.head(20)
        axes[0,0].barh(range(len(top_features)), top_features['importance'])
        axes[0,0].set_yticks(range(len(top_features)))
        axes[0,0].set_yticklabels(top_features['feature'])
        axes[0,0].set_title('Top 20 Feature Importance')
        axes[0,0].set_xlabel('Importance Score')
        axes[0,0].invert_yaxis()
        
        # 2. Actual vs Predicted (Training Data)
        axes[0,1].scatter(y_train, y_pred_train, alpha=0.6)
        axes[0,1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        axes[0,1].set_xlabel('Actual Revenue')
        axes[0,1].set_ylabel('Predicted Revenue')
        axes[0,1].set_title('Actual vs Predicted Revenue (Training Data)')
        
        # Add R² score
        r2 = r2_score(y_train, y_pred_train)
        axes[0,1].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0,1].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Residuals (Training Data)
        residuals = y_train - y_pred_train
        axes[1,0].scatter(y_pred_train, residuals, alpha=0.6)
        axes[1,0].axhline(y=0, color='r', linestyle='--')
        axes[1,0].set_xlabel('Predicted Revenue')
        axes[1,0].set_ylabel('Residuals')
        axes[1,0].set_title('Residual Plot (Training Data)')
        
        # 4. Time series for a sample store (actual vs predicted)
        # Get the clean data that was actually used for training (same as in train method)
        clean_data = self.processed_data.dropna().copy()
        sample_store = clean_data['store_id'].unique()[0]
        
        # Get data for this store from the clean training data
        store_mask = clean_data['store_id'] == sample_store
        store_data_clean = clean_data[store_mask].sort_values('date')
        
        # Get corresponding predictions (same indices as the clean data)
        store_indices = store_mask[store_mask].index
        # Map back to position in the clean data arrays
        clean_data_reset = clean_data.reset_index(drop=True)
        store_positions = clean_data_reset.index[clean_data_reset['store_id'] == sample_store]
        
        store_actual = store_data_clean['revenue'].values
        store_pred = y_pred_train[store_positions]
        store_dates = store_data_clean['date'].values
        
        axes[1,1].plot(store_dates, store_actual, label='Actual Revenue', color='blue', marker='o')
        axes[1,1].plot(store_dates, store_pred, label='Predicted Revenue', color='red', linestyle='--', marker='s')
        
        axes[1,1].set_title(f'Revenue Time Series - Store {sample_store} (Training Data)')
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel('Revenue')
        axes[1,1].legend()
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        return self

    def evaluate_model_performance(self, original_data):
        """
        Use the trained model to make predictions and update forecast_submission.csv
        Uses the same DataProcessorBuilder pipeline as training for consistency
        """
        print("🎯 Starting model evaluation and forecast generation...")
        
        # 1. Load the forecast submission template
        forecast_df = pd.read_csv('forecast_submission.csv')
        print(f"   📊 Loaded forecast template: {len(forecast_df)} predictions needed")
        
        # 2. Parse the IDs to extract store_id and date
        forecast_df['store_id'] = forecast_df['id'].str.split('_').str[0].astype(int)
        forecast_df['date_str'] = forecast_df['id'].str.split('_').str[1]
        forecast_df['date'] = pd.to_datetime(forecast_df['date_str'], format='%Y%m%d')
        
        print(f"   📅 Forecast period: {forecast_df['date'].min()} to {forecast_df['date'].max()}")
        print(f"   🏪 Stores to predict: {sorted(forecast_df['store_id'].unique())}")
        
        # 3. Merge with calendar events for the forecast period
        events = pd.read_csv('calendar_events.csv', parse_dates=['date'])
        forecast_df = forecast_df.merge(events, on='date', how='left')
        
        # 4. Add store information (country) from the original data
        store_mapping = original_data[['store_id', 'country', 'store']].drop_duplicates()
        forecast_df = forecast_df.merge(store_mapping, on='store_id', how='left')
        
        # 5. Add revenue column to forecast_df as NaN for consistency
        forecast_df['revenue'] = np.nan
        
        # 6. Combine training data with forecast data for proper lag feature calculation
        combined_data = pd.concat([
            original_data,
            forecast_df
        ]).sort_values(['store_id', 'date']).reset_index(drop=True)
        
        print("   🔧 Creating comprehensive features using same DataProcessorBuilder as training...")
        
        # 7. Use the SAME DataProcessorBuilder pipeline as training for consistency
        forecast_with_features, _ = self.data_processor.process(combined_data)
        
        # 8. Extract only the forecast rows (where revenue is NaN from our addition)
        forecast_processed = forecast_with_features[
            forecast_with_features['revenue'].isna() & 
            forecast_with_features['date'].isin(forecast_df['date'])
        ].copy()
        
        print(f"   🔧 Model was trained with {len(self.feature_columns)} features")
        print(f"   🏗️  Created features for forecasting using same pipeline")
        
        # 9. Create missing features with zeros (for unseen categories or missing features)
        for feature in self.feature_columns:
            if feature not in forecast_processed.columns:
                forecast_processed[feature] = 0
                print(f"   ➕ Added missing feature '{feature}' with default value 0")
        
        # 10. Only use features that the model was actually trained on
        X_forecast = forecast_processed[self.feature_columns].fillna(0)
        
        print(f"   ✅ Using exact same {len(self.feature_columns)} features as training")
        print(f"   🔮 Making predictions for {len(X_forecast)} data points...")
        
        # 11. Make predictions
        predictions = self.model.predict(X_forecast)
        
        # 12. Ensure predictions are non-negative (revenue can't be negative)
        predictions = np.maximum(predictions, 0)
        
        # 13. Create output dataframe with original forecast template structure
        output_df = forecast_df[['id']].copy()
        output_df['prediction'] = predictions
        
        # 14. Save to new file (preserve original template)
        output_filename = 'my_forecast_predictions.csv'
        output_df.to_csv(output_filename, index=False)
        
        print(f"   ✅ Predictions saved to {output_filename}")
        print(f"   📋 Original template preserved at forecast_submission.csv")
        print(f"   📊 Prediction statistics:")
        print(f"      Mean: ${predictions.mean():,.0f}")
        print(f"      Median: ${np.median(predictions):,.0f}")
        print(f"      Min: ${predictions.min():,.0f}")
        print(f"      Max: ${predictions.max():,.0f}")
        
        # 15. Show sample predictions
        print(f"\n   🔍 Sample predictions:")
        print(output_df.head(10).to_string(index=False))
        
        return output_df

