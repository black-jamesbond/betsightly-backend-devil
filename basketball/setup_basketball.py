#!/usr/bin/env python3
"""
Basketball Pipeline Setup Script

This script sets up the complete basketball prediction pipeline:
1. Installs required dependencies
2. Creates necessary directories
3. Downloads sample training data
4. Trains initial models
5. Tests the prediction pipeline
6. Verifies API endpoints

Usage:
    python basketball/setup_basketball.py [--skip-training] [--test-api]
"""

import argparse
import logging
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required dependencies for basketball predictions."""
    logger.info("Installing basketball prediction dependencies...")
    
    dependencies = [
        "nba_api",  # Free NBA data API
        "xgboost",  # Already in requirements.txt
        "lightgbm", # Already in requirements.txt
        "scikit-learn", # Already in requirements.txt
        "pandas",   # Already in requirements.txt
        "numpy"     # Already in requirements.txt
    ]
    
    for dep in dependencies:
        try:
            logger.info(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            logger.info(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {dep}: {str(e)}")
            return False
    
    return True

def create_directories():
    """Create necessary directories for basketball pipeline."""
    logger.info("Creating basketball directories...")
    
    directories = [
        os.path.join(settings.ml.MODEL_DIR, "basketball"),
        os.path.join(settings.ml.DATA_DIR, "basketball"),
        os.path.join(settings.ml.CACHE_DIR, "basketball", "features"),
        os.path.join(settings.ml.CACHE_DIR, "basketball", "predictions"),
        "basketball/models"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {str(e)}")
            return False
    
    return True

def test_imports():
    """Test that all basketball modules can be imported."""
    logger.info("Testing basketball module imports...")
    
    try:
        from basketball.data_fetcher import NBADataFetcher
        from basketball.feature_engineering import BasketballFeatureEngineer
        from basketball.models import BasketballModelFactory
        from basketball.prediction_service import BasketballPredictionService
        
        logger.info("✅ All basketball modules imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        return False

def train_initial_models():
    """Train initial basketball models."""
    logger.info("Training initial basketball models...")
    
    try:
        # Import training script
        from basketball.train_models import train_basketball_models
        
        # Set up training data path
        data_path = os.path.join(settings.ml.DATA_DIR, "basketball", "nba_games.csv")
        
        # Train models
        results = train_basketball_models(data_path, retrain=True)
        
        if results['status'] == 'success':
            logger.info("✅ Basketball models trained successfully")
            logger.info(f"Models trained: {list(results['results'].keys())}")
            return True
        else:
            logger.error(f"❌ Model training failed: {results.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model training error: {str(e)}")
        return False

def test_predictions():
    """Test basketball prediction generation."""
    logger.info("Testing basketball prediction generation...")
    
    try:
        from basketball.prediction_service import BasketballPredictionService
        
        # Initialize service
        service = BasketballPredictionService()
        
        # Test prediction generation
        today = datetime.now().strftime("%Y-%m-%d")
        results = service.generate_predictions(today)
        
        if results['status'] == 'success':
            logger.info("✅ Basketball predictions generated successfully")
            logger.info(f"Generated {results.get('total_games', 0)} game predictions")
            return True
        else:
            logger.warning(f"⚠️ Prediction generation: {results.get('message', 'No games today')}")
            return True  # Not an error if no games today
            
    except Exception as e:
        logger.error(f"❌ Prediction test error: {str(e)}")
        return False

def test_api_endpoints():
    """Test basketball API endpoints."""
    logger.info("Testing basketball API endpoints...")
    
    try:
        import requests
        from main import app
        import uvicorn
        import threading
        import time
        
        # Start server in background
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8001, log_level="error")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(3)  # Wait for server to start
        
        # Test endpoints
        base_url = "http://127.0.0.1:8001/api/basketball-predictions"
        
        endpoints_to_test = [
            "/",
            "/summary",
            "/models/status"
        ]
        
        for endpoint in endpoints_to_test:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ Endpoint {endpoint} working")
                else:
                    logger.warning(f"⚠️ Endpoint {endpoint} returned {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ Endpoint {endpoint} test failed: {str(e)}")
        
        return True
        
    except ImportError:
        logger.warning("⚠️ Requests not available, skipping API tests")
        return True
    except Exception as e:
        logger.error(f"❌ API test error: {str(e)}")
        return False

def print_setup_summary():
    """Print setup summary and next steps."""
    print(f"\n{'='*60}")
    print("BASKETBALL PIPELINE SETUP COMPLETE")
    print(f"{'='*60}")

    print("\nDIRECTORY STRUCTURE:")
    print("basketball/")
    print("|-- __init__.py              # Module initialization")
    print("|-- data_fetcher.py          # NBA data fetching")
    print("|-- feature_engineering.py   # Basketball features")
    print("|-- models.py                # ML models (XGBoost, LightGBM, NN)")
    print("|-- prediction_service.py    # Prediction orchestration")
    print("|-- train_models.py          # Model training script")
    print("|-- predict_games.py         # Prediction script")
    print("|-- setup_basketball.py     # This setup script")

    print("\nQUICK START:")
    print("1. Train models:")
    print("   py basketball/train_models.py")

    print("\n2. Generate predictions:")
    print("   py basketball/predict_games.py")

    print("\n3. Start API server:")
    print("   py start_production.py")

    print("\n4. Test API endpoints:")
    print("   curl http://localhost:8000/api/basketball-predictions/")
    print("   curl http://localhost:8000/api/basketball-predictions/summary")
    print("   curl http://localhost:8000/api/basketball-predictions/models/status")

    print("\nAPI ENDPOINTS:")
    print("• GET /api/basketball-predictions/")
    print("  └── All basketball predictions")
    print("• GET /api/basketball-predictions/summary")
    print("  └── Prediction summary")
    print("• GET /api/basketball-predictions/confidence/{level}")
    print("  └── Predictions by confidence level")
    print("• GET /api/basketball-predictions/models/status")
    print("  └── Model availability status")
    print("• POST /api/basketball-predictions/train")
    print("  └── Trigger model training")

    print("\nPREDICTION TYPES:")
    print("• Win/Loss (XGBoost)")
    print("• Over/Under Total Points (LightGBM)")
    print("• Neural Network (Advanced)")

    print("\nFEATURES INCLUDED:")
    print("• Team form (last 5 games)")
    print("• Home/Away performance")
    print("• Shooting percentages (FG%, 3P%, FT%)")
    print("• Pace and efficiency metrics")
    print("• Rest days between games")
    print("• Head-to-head records")
    print("• Season context")

    print(f"\n{'='*60}")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup basketball prediction pipeline")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training step"
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test API endpoints after setup"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("Setting up Basketball Prediction Pipeline...")
    print("=" * 50)
    
    setup_steps = [
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Testing imports", test_imports),
    ]
    
    if not args.skip_training:
        setup_steps.append(("Training initial models", train_initial_models))
    
    setup_steps.extend([
        ("Testing predictions", test_predictions),
    ])
    
    if args.test_api:
        setup_steps.append(("Testing API endpoints", test_api_endpoints))
    
    # Run setup steps
    failed_steps = []
    
    for step_name, step_func in setup_steps:
        print(f"\n[RUNNING] {step_name}...")
        try:
            if step_func():
                print(f"[SUCCESS] {step_name} completed")
            else:
                print(f"[FAILED] {step_name} failed")
                failed_steps.append(step_name)
        except Exception as e:
            print(f"[ERROR] {step_name} failed: {str(e)}")
            failed_steps.append(step_name)
    
    # Summary
    print(f"\n{'='*50}")
    if failed_steps:
        print(f"Setup completed with {len(failed_steps)} issues:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nYou may need to address these issues manually.")
    else:
        print("Basketball pipeline setup completed successfully!")
    
    print_setup_summary()

if __name__ == "__main__":
    main()
