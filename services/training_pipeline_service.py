"""
Training Pipeline Service

This service manages model training, continuous learning, and performance monitoring
for the BetSightly ML system.
"""

import logging
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from pathlib import Path

from database import SessionLocal
from models.training_models import ModelTrainingRun, PredictionAccuracy
from utils.common import setup_logging

# Set up logging
logger = setup_logging(__name__)

class TrainingPipelineService:
    """
    Manages the complete training pipeline for continuous learning.
    
    Features:
    - Automated model retraining
    - Performance monitoring
    - Continuous learning from new data
    - Model deployment management
    - Training scheduling
    """
    
    def __init__(self):
        """Initialize the training pipeline service."""
        # Only use entry points which actually exist in this repository.
        self.training_scripts = {
            'streamlined': 'scripts/retrain_models.py',
            'quick': 'scripts/retrain_models_fast.py',
        }
        self.root = Path(__file__).resolve().parents[1]
        self._last_candidate_dir = None
        
        self.accuracy_threshold = 0.05  # Trigger retraining if accuracy drops by 5%
        self.min_new_matches = 50       # Minimum new matches before retraining (weekly)
        self.max_training_time = 7200   # Maximum training time in seconds (2 hours for weekly)
        self.training_frequency_days = 7 # Train weekly (every 7 days)
        
    def trigger_training(self, training_type: str = 'manual', 
                        trigger_reason: str = 'manual_request',
                        force_retrain: bool = False) -> Dict[str, Any]:
        """
        Trigger model training pipeline.
        
        Args:
            training_type: Type of training ('manual', 'scheduled', 'triggered')
            trigger_reason: Reason for training
            force_retrain: Force retraining even if not needed
            
        Returns:
            Dictionary with training results
        """
        start_time = datetime.now()
        training_date = start_time.strftime("%Y-%m-%d")
        
        db = SessionLocal()
        
        try:
            # Check if training is already running
            running_training = db.query(ModelTrainingRun).filter(
                ModelTrainingRun.status == 'running'
            ).first()
            
            if running_training:
                return {
                    'status': 'error',
                    'message': 'Training already in progress',
                    'running_training_id': running_training.id
                }
            
            # Create training run record
            training_run = ModelTrainingRun(
                training_date=training_date,
                training_type=training_type,
                trigger_reason=trigger_reason,
                status='running',
                started_at=start_time
            )
            db.add(training_run)
            db.commit()
            
            logger.info(f"🚀 Starting training pipeline (ID: {training_run.id})")
            logger.info(f"📋 Type: {training_type}, Reason: {trigger_reason}")
            
            # Check if training is needed (unless forced)
            if not force_retrain and not self._is_training_needed(db):
                training_run.status = 'completed'
                training_run.deployment_status = 'rejected'
                training_run.deployment_reason = 'Training not needed'
                training_run.completed_at = datetime.now()
                db.commit()
                
                return {
                    'status': 'success',
                    'message': 'Training not needed - models are performing well',
                    'training_run_id': training_run.id
                }
            
            # Get current model performance
            previous_accuracy = self._get_current_model_accuracy(db)
            training_run.previous_accuracy = previous_accuracy
            db.commit()
            
            # Execute training
            training_results = self._execute_training(training_run.id, force_retrain)
            
            # Update training run with results
            training_time = (datetime.now() - start_time).total_seconds()
            
            training_run.training_time_seconds = training_time
            training_run.models_trained = training_results.get('models_trained', [])
            training_run.training_algorithms = training_results.get('algorithms_used', [])
            training_run.training_data_size = training_results.get('training_data_size', 0)
            training_run.new_matches_added = training_results.get('new_matches_added', 0)
            training_run.data_source = training_results.get('data_source', 'unknown')
            
            if training_results.get('status') == 'success':
                # Evaluate new models
                new_accuracy = self._evaluate_new_models()
                training_run.new_accuracy = new_accuracy
                
                # Calculate performance improvement
                performance_improvement = self._calculate_performance_improvement(
                    previous_accuracy, new_accuracy
                )
                training_run.performance_improvement = performance_improvement
                
                # Decide on deployment
                should_deploy = self._should_deploy_models(performance_improvement)
                
                if should_deploy:
                    deployment_result = self._deploy_new_models()
                    training_run.deployment_status = 'deployed' if deployment_result else 'failed'
                    training_run.deployment_reason = 'Performance improved' if deployment_result else 'Deployment failed'
                    training_run.deployed_at = datetime.now() if deployment_result else None
                else:
                    training_run.deployment_status = 'awaiting_review'
                    training_run.deployment_reason = 'Staged candidate; requires independent temporal backtest and approval'
                
                training_run.status = 'completed'
                
            else:
                training_run.status = 'failed'
                training_run.error_message = training_results.get('error', 'Unknown error')
            
            training_run.completed_at = datetime.now()
            db.commit()
            
            logger.info(f"✅ Training pipeline completed (ID: {training_run.id})")
            
            return {
                'status': 'success',
                'training_run_id': training_run.id,
                'training_time_seconds': training_time,
                'models_trained': training_run.models_trained,
                'deployment_status': training_run.deployment_status,
                'performance_improvement': training_run.performance_improvement,
                'message': f'Training completed - {training_run.deployment_status}'
            }
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            
            if 'training_run' in locals():
                training_run.status = 'failed'
                training_run.error_message = str(e)
                training_run.completed_at = datetime.now()
                db.commit()
            
            return {
                'status': 'error',
                'error': str(e),
                'training_run_id': training_run.id if 'training_run' in locals() else None
            }
            
        finally:
            db.close()
    
    def _is_training_needed(self, db: Session) -> bool:
        """Check if training is needed based on performance metrics."""
        try:
            # Check recent accuracy
            recent_accuracy = self._get_recent_accuracy(db)
            
            if not recent_accuracy:
                logger.info("No recent accuracy data - training recommended")
                return True
            
            # Check if accuracy has dropped significantly
            baseline_accuracy = self._get_baseline_accuracy(db)
            
            if baseline_accuracy and recent_accuracy < (baseline_accuracy - self.accuracy_threshold):
                logger.info(f"Accuracy dropped from {baseline_accuracy:.3f} to {recent_accuracy:.3f} - training needed")
                return True
            
            # Check if enough new data is available
            new_matches_count = self._count_new_matches(db)
            
            if new_matches_count >= self.min_new_matches:
                logger.info(f"Found {new_matches_count} new matches - training recommended")
                return True
            
            # Check time since last training (weekly schedule)
            last_training = db.query(ModelTrainingRun).filter(
                ModelTrainingRun.status == 'completed',
                ModelTrainingRun.deployment_status == 'deployed'
            ).order_by(ModelTrainingRun.completed_at.desc()).first()

            if last_training:
                days_since_training = (datetime.now() - last_training.completed_at).days
                if days_since_training >= self.training_frequency_days:  # Weekly retraining
                    logger.info(f"Last training was {days_since_training} days ago - weekly training needed")
                    return True
            else:
                logger.info("No previous successful training found - initial training needed")
                return True
            
            logger.info("Training not needed - models are performing well")
            return False
            
        except Exception as e:
            logger.error(f"Error checking if training is needed: {str(e)}")
            return True  # Default to training if we can't determine
    
    def _execute_training(self, training_run_id: int, force_retrain: bool = False) -> Dict[str, Any]:
        """Train into a per-run staging directory; never alter live model files."""
        script = self.root / self.training_scripts['streamlined']
        if not script.is_file():
            return {'status': 'error', 'error': 'training entry point missing'}
        candidate = self.root / 'models' / 'staging' / ('run_' + str(training_run_id))
        candidate.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env['BETSIGHTLY_MODELS_OUTPUT_DIR'] = str(candidate)
        self._last_candidate_dir = candidate
        try:
            result = subprocess.run(
                [sys.executable, str(script)], cwd=str(self.root), env=env,
                capture_output=True, text=True, timeout=self.max_training_time,
            )
        except subprocess.TimeoutExpired:
            return {'status': 'error', 'error': 'training timed out'}
        except OSError as exc:
            return {'status': 'error', 'error': str(exc)}
        if result.returncode != 0:
            return {'status': 'error', 'error': (result.stderr or result.stdout)[-800:]}
        try:
            meta = json.loads((candidate / 'meta.json').read_text(encoding='utf-8'))
            weights = json.loads((candidate / 'model_weights.json').read_text(encoding='utf-8'))
            if not meta.get('models') or not weights:
                raise ValueError('missing trained models or measured test scores')
        except (OSError, ValueError, TypeError) as exc:
            return {'status': 'error', 'error': 'training outputs invalid: ' + str(exc)}
        return {
            'status': 'success', 'models_trained': meta['models'],
            'algorithms_used': sorted(set(str(x).rsplit('_', 1)[-1]
                                          for x in meta['models'])),
            'training_data_size': int(meta.get('n_samples') or 0),
            'new_matches_added': 0,
            'data_source': 'api-football-history',
            'staged_path': str(candidate),
        }

    def _parse_training_output(self, output: str) -> Dict[str, Any]:
        """Parse training script output to extract results."""
        results = {
            'models_trained': [],
            'algorithms_used': [],
            'training_data_size': 0,
            'new_matches_added': 0,
            'data_source': 'github_dataset'
        }
        
        try:
            lines = output.split('\n')
            
            for line in lines:
                if 'models trained:' in line.lower():
                    # Extract number of models
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            count = int(parts[1].strip())
                            results['models_trained'] = [f'model_{i}' for i in range(count)]
                        except ValueError:
                            pass
                
                elif 'training data:' in line.lower():
                    # Extract training data size
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            size = int(parts[1].strip().split()[0])
                            results['training_data_size'] = size
                        except (ValueError, IndexError):
                            pass
                
                elif 'xgboost' in line.lower():
                    if 'xgboost' not in results['algorithms_used']:
                        results['algorithms_used'].append('xgboost')
                
                elif 'lightgbm' in line.lower():
                    if 'lightgbm' not in results['algorithms_used']:
                        results['algorithms_used'].append('lightgbm')
                
                elif 'ensemble' in line.lower():
                    if 'ensemble' not in results['algorithms_used']:
                        results['algorithms_used'].append('ensemble')
            
        except Exception as e:
            logger.warning(f"Error parsing training output: {str(e)}")
        
        return results
    
    def _get_current_model_accuracy(self, db: Session) -> Dict[str, float]:
        """Get current model accuracy metrics."""
        try:
            # This would typically query recent prediction accuracy
            # For now, return placeholder values
            return {
                'match_result': 0.65,
                'over_under': 0.62,
                'btts': 0.58,
                'overall': 0.62
            }
        except Exception as e:
            logger.error(f"Error getting current model accuracy: {str(e)}")
            return {}
    
    def _get_recent_accuracy(self, db: Session) -> Optional[float]:
        """Get recent prediction accuracy."""
        try:
            # Calculate accuracy from last 30 days
            cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            accuracy_records = db.query(PredictionAccuracy).filter(
                PredictionAccuracy.match_date >= cutoff_date,
                PredictionAccuracy.is_correct.isnot(None)
            ).all()
            
            if not accuracy_records:
                return None
            
            correct_predictions = sum(1 for record in accuracy_records if record.is_correct)
            total_predictions = len(accuracy_records)
            
            return correct_predictions / total_predictions if total_predictions > 0 else None
            
        except Exception as e:
            logger.error(f"Error getting recent accuracy: {str(e)}")
            return None
    
    def _get_baseline_accuracy(self, db: Session) -> Optional[float]:
        """Get baseline accuracy for comparison."""
        try:
            # Get accuracy from 30-60 days ago as baseline
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
            end_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            accuracy_records = db.query(PredictionAccuracy).filter(
                PredictionAccuracy.match_date >= start_date,
                PredictionAccuracy.match_date < end_date,
                PredictionAccuracy.is_correct.isnot(None)
            ).all()
            
            if not accuracy_records:
                return None
            
            correct_predictions = sum(1 for record in accuracy_records if record.is_correct)
            total_predictions = len(accuracy_records)
            
            return correct_predictions / total_predictions if total_predictions > 0 else None
            
        except Exception as e:
            logger.error(f"Error getting baseline accuracy: {str(e)}")
            return None
    
    def _count_new_matches(self, db: Session) -> int:
        """Count labeled rows newer than the active model's recorded train date."""
        path = self.root / 'data' / 'api-football' / 'matches.csv'
        meta_path = self.root / 'models' / 'api_football' / 'meta.json'
        if not path.is_file():
            return 0
        try:
            import pandas as pd
            dates = pd.to_datetime(pd.read_csv(path, usecols=['date'])['date'],
                                   errors='coerce', utc=True)
            if not meta_path.is_file():
                return int(dates.notna().sum())
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            cutoff = pd.Timestamp(meta['trained_at'])
            if cutoff.tzinfo is None:
                cutoff = cutoff.tz_localize('UTC')
            return int((dates > cutoff).sum())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning('Training data freshness unknown: %s', exc)
            return 0

    def _evaluate_new_models(self) -> Dict[str, float]:
        """Read measured holdout accuracies; never fabricate model metrics."""
        if self._last_candidate_dir is None:
            return {}
        try:
            weights = json.loads((self._last_candidate_dir / 'model_weights.json').read_text())
            targets = ('match_result', 'over_1_5', 'over_2_5', 'btts')
            scores = {}
            for target in targets:
                values = [float(v) for k, v in weights.items() if k.startswith(target + '_')]
                if not values or any(v < 0 or v > 1 for v in values):
                    return {}
                scores[target] = max(values)
            scores['overall'] = sum(scores.values()) / len(scores)
            return scores
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            logger.warning('Unable to evaluate staged model weights: %s', exc)
            return {}

    def _calculate_performance_improvement(self, previous: Dict[str, float], 
                                         new: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance improvement metrics."""
        improvement = {}
        
        for metric in previous.keys():
            if metric in new:
                improvement[metric] = new[metric] - previous[metric]
        
        return improvement
    
    def _should_deploy_models(self, performance_improvement: Dict[str, float]) -> bool:
        """No automatic promotion from a random holdout; temporal test required."""
        return False

    def _deploy_new_models(self) -> bool:
        """Production promotion must be an explicit, independently evaluated step."""
        logger.warning('Automatic promotion disabled: staged model needs temporal backtest')
        return False

# Create singleton instance
training_pipeline_service = TrainingPipelineService()
