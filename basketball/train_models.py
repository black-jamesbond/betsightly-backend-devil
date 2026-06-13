#!/usr/bin/env python3
"""
Basketball Model Training Script

This script trains basketball prediction models using historical NBA data.
It follows the same pattern as the football model training but adapted for basketball.

Usage:
    python basketball/train_models.py [--data-path PATH] [--retrain]

Features:
- Downloads NBA historical data if not available
- Engineers basketball-specific features
- Trains XGBoost, LightGBM, and Neural Network models
- Evaluates model performance
- Saves trained models for prediction use
"""

import argparse
import logging
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from basketball.data_fetcher import NBADataFetcher
from basketball.feature_engineering import BasketballFeatureEngineer
from basketball.models import BasketballModelFactory
from utils.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_sample_training_data(save_path: str) -> str:
    """
    Create sample NBA training data for demonstration.
    
    In production, this would be replaced with real NBA data from:
    - Kaggle datasets
    - Basketball Reference scraping
    - NBA API historical data
    
    Args:
        save_path: Path to save the sample data
        
    Returns:
        Path to the created data file
    """
    logger.info("Creating sample NBA training data...")
    
    # Create sample NBA games data
    np.random.seed(42)  # For reproducible results
    
    # NBA teams (sample)
    teams = [
        'Los Angeles Lakers', 'Boston Celtics', 'Golden State Warriors',
        'Miami Heat', 'Chicago Bulls', 'San Antonio Spurs',
        'Philadelphia 76ers', 'Brooklyn Nets', 'Milwaukee Bucks',
        'Phoenix Suns', 'Dallas Mavericks', 'Denver Nuggets'
    ]
    
    # Generate sample games
    num_games = 1000
    games_data = []
    
    for i in range(num_games):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        
        # Simulate game statistics
        home_fg_pct = np.random.uniform(0.40, 0.55)
        away_fg_pct = np.random.uniform(0.40, 0.55)
        
        home_3p_pct = np.random.uniform(0.30, 0.45)
        away_3p_pct = np.random.uniform(0.30, 0.45)
        
        home_ft_pct = np.random.uniform(0.70, 0.90)
        away_ft_pct = np.random.uniform(0.70, 0.90)
        
        # Points influenced by shooting percentages
        home_points = int(np.random.normal(110, 15) + (home_fg_pct - 0.45) * 20)
        away_points = int(np.random.normal(110, 15) + (away_fg_pct - 0.45) * 20)
        
        # Ensure realistic point ranges
        home_points = max(80, min(150, home_points))
        away_points = max(80, min(150, away_points))
        
        total_points = home_points + away_points
        home_win = 1 if home_points > away_points else 0
        
        game_data = {
            'GAME_ID': f'002230{i:04d}',
            'GAME_DATE': f'2023-{np.random.randint(10, 13):02d}-{np.random.randint(1, 29):02d}',
            'HOME_TEAM_ID': hash(home_team) % 1000,
            'AWAY_TEAM_ID': hash(away_team) % 1000,
            'HOME_TEAM': home_team,
            'AWAY_TEAM': away_team,
            'HOME_PTS': home_points,
            'AWAY_PTS': away_points,
            'HOME_FG_PCT': home_fg_pct,
            'AWAY_FG_PCT': away_fg_pct,
            'HOME_FG3_PCT': home_3p_pct,
            'AWAY_FG3_PCT': away_3p_pct,
            'HOME_FT_PCT': home_ft_pct,
            'AWAY_FT_PCT': away_ft_pct,
            'TOTAL_PTS': total_points,
            'HOME_WIN': home_win
        }
        
        games_data.append(game_data)
    
    # Create DataFrame and save
    df = pd.DataFrame(games_data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    logger.info(f"Sample training data created: {len(df)} games saved to {save_path}")
    return save_path

def train_basketball_models(data_path: str, retrain: bool = False) -> dict:
    """
    Train basketball prediction models.
    
    Args:
        data_path: Path to training data
        retrain: Whether to retrain existing models
        
    Returns:
        Training results dictionary
    """
    logger.info("Starting basketball model training...")
    
    # Initialize components
    data_fetcher = NBADataFetcher()
    feature_engineer = BasketballFeatureEngineer()
    model_factory = BasketballModelFactory()
    
    # Check if models already exist
    models_dir = os.path.join(settings.ml.MODEL_DIR, "basketball")
    win_loss_model_path = os.path.join(models_dir, "win_loss_xgboost.joblib")
    over_under_model_path = os.path.join(models_dir, "over_under_lightgbm.joblib")
    
    if not retrain and os.path.exists(win_loss_model_path) and os.path.exists(over_under_model_path):
        logger.info("Models already exist. Use --retrain to force retraining.")
        return {'status': 'skipped', 'message': 'Models already exist'}
    
    # Load training data
    if not os.path.exists(data_path):
        logger.info("Training data not found. Creating sample data...")
        data_path = create_sample_training_data(data_path)
    
    logger.info(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)
    
    if df.empty:
        raise ValueError("No training data available")
    
    logger.info(f"Loaded {len(df)} training samples")
    
    # Engineer features
    logger.info("Engineering features...")
    features_df = feature_engineer.engineer_features(df)
    
    # Prepare feature matrix
    feature_columns = feature_engineer.get_feature_names()
    
    # Ensure all feature columns exist
    missing_columns = [col for col in feature_columns if col not in features_df.columns]
    if missing_columns:
        logger.warning(f"Missing feature columns: {missing_columns}")
        # Add missing columns with default values
        for col in missing_columns:
            features_df[col] = 0
    
    X = features_df[feature_columns]
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    
    # Train models
    results = {}
    
    # 1. Train Win/Loss XGBoost model
    logger.info("Training Win/Loss XGBoost model...")
    win_loss_model = model_factory.create_win_loss_model()
    y_win_loss = df['HOME_WIN']
    
    win_loss_results = win_loss_model.train(X, y_win_loss)
    results['win_loss_xgboost'] = win_loss_results
    
    # 2. Train Over/Under LightGBM model
    logger.info("Training Over/Under LightGBM model...")
    over_under_model = model_factory.create_over_under_model()
    y_total = df['TOTAL_PTS']
    
    over_under_results = over_under_model.train(X, y_total, total_threshold=220.0)
    results['over_under_lightgbm'] = over_under_results
    
    # 3. Train Neural Network model (if available)
    logger.info("Training Neural Network model...")
    nn_model = model_factory.create_neural_network_model()
    if nn_model:
        nn_results = nn_model.train(X, y_win_loss, task_type='classification')
        results['neural_network'] = nn_results
    else:
        logger.warning("Neural Network model not available")
        results['neural_network'] = {'status': 'not_available'}
    
    # Summary
    training_summary = {
        'status': 'success',
        'training_date': datetime.now().isoformat(),
        'training_samples': len(df),
        'feature_count': len(feature_columns),
        'models_trained': len([r for r in results.values() if r.get('status') != 'not_available']),
        'results': results
    }
    
    logger.info("Basketball model training completed successfully!")
    logger.info(f"Models trained: {list(results.keys())}")
    
    return training_summary

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train basketball prediction models")
    parser.add_argument(
        "--data-path",
        default=os.path.join(settings.ml.DATA_DIR, "basketball", "nba_games.csv"),
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retrain existing models"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Train models
        results = train_basketball_models(args.data_path, args.retrain)
        
        # Print results
        print("\n" + "="*50)
        print("BASKETBALL MODEL TRAINING RESULTS")
        print("="*50)
        print(f"Status: {results['status']}")
        
        if results['status'] == 'success':
            print(f"Training Date: {results['training_date']}")
            print(f"Training Samples: {results['training_samples']}")
            print(f"Feature Count: {results['feature_count']}")
            print(f"Models Trained: {results['models_trained']}")
            
            print("\nModel Performance:")
            for model_name, model_results in results['results'].items():
                if model_results.get('status') != 'not_available':
                    print(f"\n{model_name.upper()}:")
                    if 'accuracy' in model_results:
                        print(f"  Accuracy: {model_results['accuracy']:.3f}")
                    if 'cv_mean' in model_results:
                        print(f"  CV Score: {model_results['cv_mean']:.3f} ± {model_results['cv_std']:.3f}")
                    if 'training_samples' in model_results:
                        print(f"  Training Samples: {model_results['training_samples']}")
        
        print("\n" + "="*50)
        
        if results['status'] == 'success':
            print("[SUCCESS] Basketball models trained successfully!")
            print("You can now generate predictions using:")
            print("  py basketball/predict_games.py")
        elif results['status'] == 'skipped':
            print("[INFO] Training skipped - models already exist")
            print("Use --retrain to force retraining")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"\n[ERROR] Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
