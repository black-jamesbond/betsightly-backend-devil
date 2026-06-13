#!/usr/bin/env python3
"""
Basketball Game Prediction Script

This script generates basketball predictions for NBA games using trained models.
It follows the same pattern as the football prediction script but adapted for basketball.

Usage:
    python basketball/predict_games.py [--date YYYY-MM-DD] [--output json|console]

Features:
- Fetches today's NBA games (or specified date)
- Generates Win/Loss and Over/Under predictions
- Provides confidence scores and explanations
- Outputs predictions in JSON format or console display
"""

import argparse
import logging
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from basketball.prediction_service import BasketballPredictionService
from utils.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def format_prediction_console(prediction: dict) -> str:
    """
    Format a single prediction for console display.
    
    Args:
        prediction: Prediction dictionary
        
    Returns:
        Formatted string for console output
    """
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"{prediction['away_team']} @ {prediction['home_team']}")
    output.append(f"Date: {prediction['game_date']}")
    output.append(f"Overall Confidence: {prediction['overall_confidence']:.1%}")
    output.append(f"{'='*60}")

    predictions = prediction.get('predictions', {})

    # Win/Loss prediction
    if 'win_loss' in predictions:
        wl = predictions['win_loss']
        output.append(f"\nWIN/LOSS PREDICTION:")
        output.append(f"   Prediction: {wl['prediction']}")
        output.append(f"   Home Win: {wl['home_win_probability']:.1%}")
        output.append(f"   Away Win: {wl['away_win_probability']:.1%}")
        output.append(f"   Confidence: {wl['confidence']:.1%}")

    # Over/Under prediction
    if 'over_under' in predictions:
        ou = predictions['over_under']
        output.append(f"\nOVER/UNDER PREDICTION:")
        output.append(f"   Prediction: {ou['prediction']} {ou['threshold']}")
        output.append(f"   Over: {ou['over_probability']:.1%}")
        output.append(f"   Under: {ou['under_probability']:.1%}")
        output.append(f"   Confidence: {ou['confidence']:.1%}")

    # Neural Network prediction (if available)
    if 'neural_network' in predictions:
        nn = predictions['neural_network']
        output.append(f"\nNEURAL NETWORK:")
        output.append(f"   Prediction: {nn.get('prediction', 'N/A')}")
        if 'confidence' in nn:
            output.append(f"   Confidence: {nn['confidence']:.1%}")
    
    return '\n'.join(output)

def format_predictions_json(predictions_data: dict) -> str:
    """
    Format predictions as JSON.
    
    Args:
        predictions_data: Predictions data dictionary
        
    Returns:
        JSON formatted string
    """
    # Create a clean JSON structure
    json_output = {
        'date': predictions_data['date'],
        'status': predictions_data['status'],
        'total_games': predictions_data.get('total_games', 0),
        'generated_at': datetime.now().isoformat(),
        'predictions': []
    }
    
    for pred in predictions_data.get('predictions', []):
        json_pred = {
            'game_id': pred['game_id'],
            'game_date': pred['game_date'],
            'matchup': f"{pred['away_team']} @ {pred['home_team']}",
            'home_team': pred['home_team'],
            'away_team': pred['away_team'],
            'overall_confidence': round(pred['overall_confidence'], 3),
            'predictions': {}
        }
        
        # Add individual predictions
        for pred_type, pred_data in pred.get('predictions', {}).items():
            if isinstance(pred_data, dict):
                json_pred['predictions'][pred_type] = {
                    k: round(v, 3) if isinstance(v, float) else v
                    for k, v in pred_data.items()
                }
        
        json_output['predictions'].append(json_pred)
    
    # Add categories if available
    if 'categories' in predictions_data:
        json_output['categories'] = {
            category: len(games) 
            for category, games in predictions_data['categories'].items()
        }
    
    # Add models status
    if 'models_used' in predictions_data:
        json_output['models_status'] = predictions_data['models_used']
    
    return json.dumps(json_output, indent=2, default=str)

def display_summary(predictions_data: dict):
    """Display prediction summary."""
    print(f"\n{'='*60}")
    print(f"NBA PREDICTIONS SUMMARY - {predictions_data['date']}")
    print(f"{'='*60}")

    if predictions_data['status'] != 'success':
        print(f"[ERROR] Status: {predictions_data['status']}")
        if 'message' in predictions_data:
            print(f"Message: {predictions_data['message']}")
        return

    total_games = predictions_data.get('total_games', 0)
    print(f"Total Games: {total_games}")

    if total_games == 0:
        print("[INFO] No NBA games scheduled for this date")
        return

    # Show categories if available
    categories = predictions_data.get('categories', {})
    if categories:
        print(f"\nCONFIDENCE BREAKDOWN:")
        for category, games in categories.items():
            print(f"   {category.replace('_', ' ').title()}: {len(games)} games")

    # Show models status
    models_status = predictions_data.get('models_used', {})
    if models_status:
        print(f"\nMODELS STATUS:")
        for model, available in models_status.items():
            status = "[OK]" if available else "[FAIL]"
            print(f"   {status} {model.replace('_', ' ').title()}")

    print(f"\n{'='*60}")

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Generate basketball predictions")
    parser.add_argument(
        "--date",
        help="Date for predictions (YYYY-MM-DD). Default: today"
    )
    parser.add_argument(
        "--output",
        choices=['json', 'console'],
        default='console',
        help="Output format (default: console)"
    )
    parser.add_argument(
        "--save",
        help="Save predictions to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Use provided date or default to today
    prediction_date = args.date or datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Initialize prediction service
        logger.info("Initializing basketball prediction service...")
        prediction_service = BasketballPredictionService()
        
        # Generate predictions
        logger.info(f"Generating predictions for {prediction_date}...")
        predictions_data = prediction_service.generate_predictions(prediction_date)
        
        # Display summary
        display_summary(predictions_data)
        
        if predictions_data['status'] != 'success' or predictions_data.get('total_games', 0) == 0:
            return
        
        # Output predictions
        if args.output == 'json':
            json_output = format_predictions_json(predictions_data)
            print(f"\n{json_output}")
            
            # Save to file if requested
            if args.save:
                with open(args.save, 'w') as f:
                    f.write(json_output)
                print(f"\n[SAVED] Predictions saved to {args.save}")
        
        else:  # console output
            predictions = predictions_data.get('predictions', [])
            
            # Sort by confidence
            predictions.sort(key=lambda x: x.get('overall_confidence', 0), reverse=True)
            
            for prediction in predictions:
                print(format_prediction_console(prediction))
            
            # Save to file if requested
            if args.save:
                json_output = format_predictions_json(predictions_data)
                with open(args.save, 'w') as f:
                    f.write(json_output)
                print(f"\n[SAVED] Predictions also saved to {args.save}")
        
        print(f"\n[SUCCESS] Basketball predictions generated successfully!")

    except Exception as e:
        logger.error(f"Prediction generation failed: {str(e)}")
        print(f"\n[ERROR] Prediction failed: {str(e)}")

        # Check if models are trained
        models_dir = Path(settings.ml.MODEL_DIR) / "basketball"
        if not models_dir.exists() or not any(models_dir.glob("*.joblib")):
            print("\n[TIP] Train models first using:")
            print("   py basketball/train_models.py")

        sys.exit(1)

if __name__ == "__main__":
    main()
