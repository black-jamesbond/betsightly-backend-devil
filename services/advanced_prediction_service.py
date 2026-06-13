#!/usr/bin/env python3
"""
Advanced Prediction Service - Production ML Integration

This service integrates all advanced ML models (XGBoost, Ensemble, Enhanced)
for maximum prediction accuracy and sophistication.

Features:
- XGBoost models with hyperparameter optimization
- Ensemble models with voting and stacking
- Advanced feature engineering pipeline
- SHAP/LIME explanations
- Meta-model stacking
- Calibrated confidence scores
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import requests
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Import ML components with robust error handling
try:
    from ml.feature_engineering import AdvancedFootballFeatureEngineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced feature engineering not available: {str(e)}")
    FEATURE_ENGINEERING_AVAILABLE = False

try:
    from ml.model_factory import ModelFactory
    MODEL_FACTORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Model factory not available: {str(e)}")
    MODEL_FACTORY_AVAILABLE = False

# Import model compatibility service
try:
    from services.model_compatibility_service import model_compatibility_service
    MODEL_COMPATIBILITY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Model compatibility service not available: {str(e)}")
    MODEL_COMPATIBILITY_AVAILABLE = False

ADVANCED_ML_AVAILABLE = FEATURE_ENGINEERING_AVAILABLE and MODEL_FACTORY_AVAILABLE

# --- New statistical models (module-level singletons) ---
try:
    from ml.elo_model import EloRatingSystem
    _elo_system = EloRatingSystem()
    ELO_AVAILABLE = True
except ImportError:
    _elo_system = None
    ELO_AVAILABLE = False

try:
    from ml.dixon_coles_model import DixonColesModel
    _dixon_coles = DixonColesModel()
    DIXON_COLES_AVAILABLE = True
except ImportError:
    _dixon_coles = None
    DIXON_COLES_AVAILABLE = False

try:
    from ml.bayesian_averaging import BayesianModelAverager
    _bma = BayesianModelAverager()
    BMA_AVAILABLE = True
except ImportError:
    _bma = None
    BMA_AVAILABLE = False


# Disable SHAP temporarily for memory optimization on Render
SHAP_AVAILABLE = False
logger.info("ℹ️ SHAP disabled for memory optimization")

# Disable LIME temporarily for memory optimization
LIME_AVAILABLE = False
logger.info("ℹ️ LIME disabled for memory optimization")


class AdvancedPredictionService:
    """
    Advanced ML prediction service using XGBoost, ensemble models, and explanations.
    
    This service provides the highest accuracy predictions by combining:
    - Pre-trained XGBoost models
    - Ensemble models with voting
    - Advanced feature engineering
    - Model explanations (SHAP/LIME)
    - Meta-model stacking
    """
    
    def __init__(self):
        """Initialize the advanced prediction service (models are loaded lazily on first use)."""
        self.football_data_api_key = os.getenv("FOOTBALL_DATA_API_KEY")
        self.api_football_key = os.getenv("API_FOOTBALL_API_KEY")
        self.base_url_football_data = "https://api.football-data.org/v4"
        self.base_url_api_football = "https://v3.football.api-sports.io"

        # ML components — populated on first use via _ensure_loaded()
        self.feature_engineer = None
        self.model_factory = None
        self.models = {}
        self.explainers = {}

        # Memory optimization for Render (512MB limit)
        self.memory_limit_mb = 400  # Leave 112MB buffer
        self.models_loaded = 0
        self._loaded = False

    def _ensure_loaded(self):
        """Load ML components and models on first use (lazy init)."""
        if self._loaded:
            return
        self._initialize_ml_components()
        self._load_advanced_models()
        self._train_statistical_models()
        self._setup_explainers()
        self._loaded = True
        logger.info(f"Advanced Prediction Service loaded {self.models_loaded} models")
    
    def _initialize_ml_components(self):
        """Initialize ML components and inject historical data into feature engineer."""
        if FEATURE_ENGINEERING_AVAILABLE:
            try:
                self.feature_engineer = AdvancedFootballFeatureEngineer()
                self._load_historical_data()
                logger.info("✅ Advanced feature engineering initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize feature engineering: {str(e)}")

        if MODEL_FACTORY_AVAILABLE:
            try:
                self.model_factory = ModelFactory()
                logger.info("✅ Model factory initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize model factory: {str(e)}")

        if ADVANCED_ML_AVAILABLE:
            logger.info("✅ Advanced ML components fully initialized")
        else:
            logger.warning("⚠️ Advanced ML components partially available - using fallbacks")

    def _load_historical_data(self):
        """Load the GitHub dataset into the feature engineer."""
        try:
            from services.historical_data_service import historical_data_service
            df = historical_data_service.get_historical_data()
            if df is not None and self.feature_engineer is not None:
                self.feature_engineer.set_historical_data(df)
                logger.info(f"✅ Historical data injected: {len(df):,} matches available for feature engineering")
                self._historical_data_service = historical_data_service
            else:
                logger.warning("⚠️ Historical data not available — feature engineering will use heuristics")
        except Exception as e:
            logger.error(f"❌ Failed to load historical data: {e}")
    
    def _load_advanced_models(self):
        """Load all available advanced models, preferring API-Football retrained models."""
        # Try the API-Football retrained models first (team names align with live API)
        api_meta_path = Path("models/api_football/meta.json")
        if api_meta_path.exists():
            try:
                import json
                with open(api_meta_path) as f:
                    self._api_model_meta = json.load(f)

                self._api_models = {}
                self._api_model_weights = {}

                # Load accuracy-based weights if available
                weights_path = Path("models/api_football/model_weights.json")
                if weights_path.exists():
                    with open(weights_path) as f:
                        self._api_model_weights = json.load(f)
                    logger.info(f"  Loaded accuracy weights for {len(self._api_model_weights)} models")

                for model_name in self._api_model_meta.get("models", []):
                    p = Path(f"models/api_football/{model_name}.joblib")
                    if p.exists():
                        loaded = joblib.load(p)
                        # Neural network models are saved as (model, scaler) tuples
                        if model_name.endswith("_nn") and isinstance(loaded, tuple):
                            self._api_models[model_name] = {"model": loaded[0], "scaler": loaded[1], "is_nn": True}
                        else:
                            self._api_models[model_name] = {"model": loaded, "scaler": None, "is_nn": False}

                # Load the API-Football historical data for feature computation
                api_csv = Path("data/api-football/matches.csv")
                if api_csv.exists():
                    self._api_df = pd.read_csv(api_csv, low_memory=False)
                    self._api_df["date"] = pd.to_datetime(self._api_df["date"], errors="coerce")
                    self._api_df = self._api_df.dropna(subset=["date"]).sort_values("date")

                    nn_count = sum(1 for v in self._api_models.values() if v["is_nn"])
                    tree_count = len(self._api_models) - nn_count
                    logger.info(
                        f"✅ API-Football full ensemble loaded: {len(self._api_models)} models "
                        f"({tree_count} tree-based + {nn_count} neural nets), "
                        f"{len(self._api_df):,} historical matches"
                    )
                    return  # Use these — skip legacy models
                else:
                    logger.warning("API-Football models found but matches.csv missing — run fetch_history.py")
            except Exception as e:
                logger.warning(f"Could not load API-Football models: {e}")

        # Fall back to legacy xgboost models
        self._api_models = {}
        self._api_model_weights = {}
        self._api_df = None

        model_directories = [
            ("xgboost", "models/xgboost"),
        ]
        for model_type, model_dir in model_directories:
            self._load_models_from_directory(model_type, model_dir)

    def _load_api_football_models(self) -> None:
        """
        Public hot-reload entry point called by LeagueAdaptiveTrainer after
        new models are trained.  Re-reads models/api_football/ and
        data/api-football/matches.csv without restarting the process.
        """
        logger.info("🔄 Hot-reloading API-Football models …")
        try:
            self._load_advanced_models()
            self._train_statistical_models()
            logger.info("✅ Hot-reload complete")
        except Exception as e:
            logger.warning(f"⚠️  Hot-reload error (non-critical): {e}")

    def _train_statistical_models(self):
        """Train ELO and Dixon-Coles on historical data (called once at startup)."""
        if self._api_df is None or len(self._api_df) < 50:
            return
        try:
            matches = []
            for _, r in self._api_df.iterrows():
                try:
                    matches.append({
                        "home_team": r["home_team"],
                        "away_team": r["away_team"],
                        "home_goals": int(r["home_score"]),
                        "away_goals": int(r["away_score"]),
                        "date": str(r["date"]),
                    })
                except (ValueError, KeyError):
                    continue
            if not matches:
                return
            if ELO_AVAILABLE and _elo_system is not None:
                _elo_system.train(matches)
                logger.info(f"ELO trained on {len(matches)} historical matches")
            if DIXON_COLES_AVAILABLE and _dixon_coles is not None:
                _dixon_coles.train(matches[-2000:])
                logger.info("Dixon-Coles trained on last 2000 matches")
        except Exception as e:
            logger.warning(f"Statistical model training failed: {e}")

    def _load_models_from_directory(self, model_type: str, model_dir: str):
        """Load models from a specific directory."""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            return
        
        model_files = list(model_path.glob("*.joblib"))
        loaded_count = 0
        
        for model_file in model_files:
            # Extract model name (remove .joblib extension)
            model_name = model_file.stem
            full_model_name = f"{model_type}_{model_name}"

            if MODEL_COMPATIBILITY_AVAILABLE:
                # Use compatibility service for safe loading
                model, success = model_compatibility_service.load_model_safely(
                    str(model_file), full_model_name
                )

                if model is not None:
                    self.models[full_model_name] = {
                        "model": model,
                        "type": model_type,
                        "name": model_name,
                        "path": str(model_file),
                        "compatibility_status": "success" if success else "fallback"
                    }
                    loaded_count += 1
                    self.models_loaded += 1

            else:
                # Fallback to direct loading
                try:
                    model = joblib.load(model_file)

                    self.models[full_model_name] = {
                        "model": model,
                        "type": model_type,
                        "name": model_name,
                        "path": str(model_file),
                        "compatibility_status": "direct_load"
                    }

                    loaded_count += 1
                    logger.info(f"✅ Loaded {full_model_name}")

                except Exception as e:
                    logger.error(f"❌ Failed to load {model_file}: {str(e)}")
        
        logger.info(f"✅ Loaded {loaded_count} models from {model_dir}")
    
    def _setup_explainers(self):
        """Setup SHAP and LIME explainers for loaded models."""
        if not (SHAP_AVAILABLE or LIME_AVAILABLE):
            logger.warning("No explanation libraries available")
            return
        
        # Setup explainers for XGBoost models
        for model_name, model_info in self.models.items():
            if "xgboost" in model_name and SHAP_AVAILABLE:
                if MODEL_COMPATIBILITY_AVAILABLE:
                    # Use compatibility service for SHAP setup
                    explainer = model_compatibility_service.setup_shap_safely(
                        model_info["model"], model_name
                    )

                    if explainer is not None:
                        self.explainers[model_name] = {
                            "type": "shap",
                            "explainer": explainer
                        }
                        logger.info(f"✅ SHAP explainer ready for {model_name}")
                else:
                    # Fallback to direct SHAP setup
                    try:
                        explainer = shap.TreeExplainer(model_info["model"])
                        self.explainers[model_name] = {
                            "type": "shap",
                            "explainer": explainer
                        }
                        logger.info(f"✅ SHAP explainer ready for {model_name}")
                    except Exception as e:
                        logger.error(f"❌ Failed to setup SHAP for {model_name}: {str(e)}")
    
    def _run_league_adaptive_training(self, date_str: str) -> None:
        """
        Trigger LeagueAdaptiveTrainer for any leagues in today's fixtures that
        are new or have stale historical data.  Runs silently — prediction
        continues even if training fails.
        """
        try:
            from services.league_adaptive_training import league_adaptive_trainer
            report = league_adaptive_trainer.run_for_date(date_str)
            if report.get("leagues_fetched"):
                fetched = [lg["name"] for lg in report["leagues_fetched"]]
                logger.info(
                    f"✅ League adaptive training: fetched data for {fetched}, "
                    f"added {report['rows_added']} rows, "
                    f"retrained={report['models_retrained']}"
                )
            else:
                logger.debug("League adaptive training: all leagues up-to-date, no fetch needed")
        except Exception as e:
            logger.warning(f"⚠️  League adaptive training skipped: {e}")

    def get_predictions_for_date(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Get advanced ML predictions for a specific date.

        Args:
            date_str: Date in YYYY-MM-DD format (default: today)

        Returns:
            Dictionary with advanced predictions and metadata
        """
        self._ensure_loaded()

        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # Adaptively fetch history + retrain for any new/stale leagues in today's card
        self._run_league_adaptive_training(date_str)

        start_time = datetime.now()

        try:
            # Get fixtures from API
            fixtures = self._get_fixtures_from_api(date_str)

            if not fixtures:
                return self._empty_result(date_str, "No fixtures found")
            
            # Generate advanced predictions
            predictions = []
            for fixture in fixtures:
                prediction = self._predict_fixture_advanced(fixture)
                if prediction:
                    predictions.append(prediction)
            
            # Categorize predictions using advanced logic
            categories = self._categorize_predictions_advanced(predictions)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "success",
                "date": date_str,
                "predictions": predictions,
                "categories": categories,
                "metadata": {
                    "service": "advanced_prediction_service",
                    "models_used": list(self.models.keys()),
                    "total_fixtures": len(fixtures),
                    "total_predictions": len(predictions),
                    "processing_time_seconds": round(processing_time, 3),
                    "advanced_features": {
                        "xgboost_models": len([m for m in self.models.keys() if "xgboost" in m]),
                        "ensemble_models": len([m for m in self.models.keys() if "enhanced" in m]),
                        "explainers_available": len(self.explainers),
                        "feature_engineering": "advanced"
                    }
                },
                "data_source": "real_football_data"
            }
            
        except Exception as e:
            logger.error(f"❌ Error in advanced predictions: {str(e)}")
            return self._empty_result(date_str, str(e))
    
    def _empty_result(self, date_str: str, message: str) -> Dict[str, Any]:
        """Return empty result with error message."""
        return {
            "status": "success",
            "date": date_str,
            "predictions": [],
            "categories": {"2_odds": [], "5_odds": [], "10_odds": [], "rollover": []},
            "message": message,
            "metadata": {"service": "advanced_prediction_service", "models_loaded": len(self.models)}
        }

    def _get_fixtures_from_api(self, date_str: str) -> List[Dict]:
        """Get fixtures from API using real API keys."""
        try:
            # API-Football (api-sports.io) is primary — broad league coverage
            if self.api_football_key and len(self.api_football_key) > 10:
                logger.info("🌐 Using API-Football (api-sports.io)")
                return self._get_fixtures_api_football(date_str)

            # Football-Data.org as fallback
            elif self.football_data_api_key and len(self.football_data_api_key) > 10:
                logger.info("🌐 Using Football-Data.org API")
                return self._get_fixtures_football_data(date_str)

            else:
                logger.warning("⚠️  No valid API keys found")
                return []

        except Exception as e:
            logger.error(f"❌ Error fetching fixtures: {str(e)}")
            return []

    def _get_fixtures_football_data(self, date_str: str) -> List[Dict]:
        """Get fixtures from Football-Data.org API."""
        try:
            headers = {"X-Auth-Token": self.football_data_api_key}

            # Get fixtures for major leagues
            leagues = ["PL", "BL1", "SA", "PD", "FL1"]  # Premier League, Bundesliga, Serie A, La Liga, Ligue 1
            all_fixtures = []

            for league in leagues:
                url = f"{self.base_url_football_data}/competitions/{league}/matches"
                params = {"dateFrom": date_str, "dateTo": date_str}

                response = requests.get(url, headers=headers, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    fixtures = data.get("matches", [])
                    all_fixtures.extend(fixtures)
                    logger.info(f"✅ Got {len(fixtures)} fixtures from {league}")
                else:
                    logger.warning(f"⚠️  Failed to get {league} fixtures: {response.status_code}")

            return all_fixtures

        except Exception as e:
            logger.error(f"❌ Football-Data.org API error: {str(e)}")
            return []

    # League IDs for the competitions our models are trained on.
    # See https://www.api-football.com/documentation-v3#tag/Leagues
    _TARGET_LEAGUE_IDS = {
        39,   # England - Premier League
        40,   # England - Championship
        41,   # England - League One
        61,   # France - Ligue 1
        62,   # France - Ligue 2
        78,   # Germany - Bundesliga
        79,   # Germany - 2. Bundesliga
        135,  # Italy - Serie A
        136,  # Italy - Serie B
        140,  # Spain - La Liga
        141,  # Spain - Segunda División
        94,   # Portugal - Primeira Liga
        88,   # Netherlands - Eredivisie
        144,  # Belgium - Pro League
        203,  # Turkey - Süper Lig
        2,    # UEFA Champions League
        3,    # UEFA Europa League
        848,  # UEFA Conference League
    }

    def _get_fixtures_api_football(self, date_str: str) -> List[Dict]:
        """Get fixtures from API-Football (api-sports.io direct endpoint)."""
        try:
            headers = {"x-apisports-key": self.api_football_key}
            url = f"{self.base_url_api_football}/fixtures"
            params = {"date": date_str}

            response = requests.get(url, headers=headers, params=params, timeout=15)

            if response.status_code != 200:
                logger.warning(f"⚠️  API-Football failed: {response.status_code} — {response.text[:200]}")
                return []

            all_fixtures = response.json().get("response", [])
            # Filter to leagues our models are trained on, but fall back to
            # all fixtures when no target-league games are scheduled (e.g. int'l break)
            target_fixtures = [
                f for f in all_fixtures
                if f.get("league", {}).get("id") in self._TARGET_LEAGUE_IDS
            ]
            if target_fixtures:
                fixtures = target_fixtures
                logger.info(
                    f"✅ API-Football: {len(fixtures)} target-league fixtures "
                    f"(of {len(all_fixtures)} total) on {date_str}"
                )
            else:
                fixtures = all_fixtures
                logger.info(
                    f"✅ API-Football: no target-league fixtures — using all "
                    f"{len(fixtures)} fixtures on {date_str}"
                )
            return fixtures

        except Exception as e:
            logger.error(f"❌ API-Football error: {str(e)}")
            return []

    def _predict_fixture_advanced(self, fixture: Dict) -> Optional[Dict]:
        """Generate advanced ML prediction for a single fixture."""
        try:
            # Extract team information
            if "homeTeam" in fixture:  # Football-Data.org format
                home_team = fixture["homeTeam"]["name"]
                away_team = fixture["awayTeam"]["name"]
                league = fixture.get("competition", {}).get("name", "Unknown")
                league_tier = 1
            else:  # API-Football format
                home_team = fixture["teams"]["home"]["name"]
                away_team = fixture["teams"]["away"]["name"]
                league = fixture.get("league", {}).get("name", "Unknown")
                league_id = fixture.get("league", {}).get("id", 0)
                league_tier = 2 if league_id in {40, 62, 79, 136, 141} else 1

            # --- Use retrained API-Football models if available (perfect name alignment) ---
            if self._api_models and self._api_df is not None:
                api_features = self._compute_api_features(home_team, away_team, league_tier)
                api_result = self._predict_with_api_models(api_features, home_team, away_team)
                if api_result:
                    confidence = api_result["confidence"]
                    odds = self._calculate_odds_from_confidence(confidence)
                    category = self._determine_odds_category(odds)
                    if category is None:
                        return None
                    return {
                        "home_team":  home_team,
                        "away_team":  away_team,
                        "league":     league,
                        "bet_type":   api_result["prediction"],
                        "prediction": api_result["prediction"],
                        "confidence": confidence,
                        "odds":       odds,
                        "category":   category,
                        "model_predictions": api_result["model_predictions"],
                        "model_disagreement": api_result.get("model_disagreement", 0.0),
                        "elo_gap": api_result.get("elo_gap", 0.0),
                        "explanations": {"available": False},
                        "advanced_features": {
                            "meta_stacking": False,
                            "ensemble_voting": True,
                            "feature_engineering": "api_football_v2",
                            "models_used": len(api_result["model_predictions"]),
                            "elo_enabled": ELO_AVAILABLE,
                            "dixon_coles_enabled": DIXON_COLES_AVAILABLE and (_dixon_coles is not None and _dixon_coles.is_trained),
                            "bma_enabled": BMA_AVAILABLE,
                        },
                    }

            # --- Legacy path: old feature engineering + old models ---
            features = self._engineer_match_features_advanced(home_team, away_team, league)

            # Get predictions from all available models
            model_predictions = self._get_ensemble_predictions(features, home_team, away_team)

            # Apply meta-model stacking
            final_prediction = self._apply_meta_stacking(model_predictions, features)

            # Generate explanations if available
            explanations = self._generate_explanations(features, final_prediction)

            # Calculate advanced confidence score
            confidence = self._calculate_advanced_confidence(model_predictions, final_prediction)

            # Determine odds category
            odds = self._calculate_odds_from_confidence(confidence)
            category = self._determine_odds_category(odds)

            raw_label = final_prediction["prediction"]
            bet_type = self._resolve_bet_label(raw_label, model_predictions, home_team, away_team)

            return {
                "home_team": home_team,
                "away_team": away_team,
                "league": league,
                "bet_type": bet_type,
                "prediction": bet_type,
                "confidence": confidence,
                "odds": odds,
                "category": category,
                "model_predictions": model_predictions,
                "explanations": explanations,
                "advanced_features": {
                    "meta_stacking": True,
                    "ensemble_voting": True,
                    "feature_engineering": "advanced",
                    "models_used": len(model_predictions)
                }
            }

        except Exception as e:
            logger.error(f"❌ Error predicting fixture: {str(e)}")
            return None

    # ------------------------------------------------------------------
    # API-Football feature engineering (mirrors retrain_models.py exactly)
    # ------------------------------------------------------------------

    def _compute_api_features(self, home: str, away: str, league_tier: int = 1) -> Optional[list]:
        """Compute feature vector using API-Football historical data."""
        if self._api_df is None or len(self._api_df) == 0:
            return None

        now = datetime.now()
        df_past = self._api_df[self._api_df["date"] < pd.Timestamp(now)]

        def team_stats(team, n):
            games = df_past[
                (df_past["home_team"] == team) | (df_past["away_team"] == team)
            ].tail(n)
            if len(games) == 0:
                return {"win_rate": 0.5, "draw_rate": 0.25, "goals_scored": 1.2, "goals_conceded": 1.2}
            wins = draws = gs = gc = 0
            for _, r in games.iterrows():
                if r["home_team"] == team:
                    s, c = r["home_score"], r["away_score"]
                else:
                    s, c = r["away_score"], r["home_score"]
                gs += s; gc += c
                if s > c: wins += 1
                elif s == c: draws += 1
            n_g = len(games)
            return {"win_rate": wins/n_g, "draw_rate": draws/n_g,
                    "goals_scored": gs/n_g, "goals_conceded": gc/n_g}

        def home_stats(team, n):
            games = df_past[df_past["home_team"] == team].tail(n)
            if len(games) == 0:
                return {"win_rate": 0.5, "goals_scored": 1.5}
            wins  = sum(1 for _, r in games.iterrows() if r["home_score"] > r["away_score"])
            goals = sum(r["home_score"] for _, r in games.iterrows())
            return {"win_rate": wins/len(games), "goals_scored": goals/len(games)}

        def away_stats(team, n):
            games = df_past[df_past["away_team"] == team].tail(n)
            if len(games) == 0:
                return {"win_rate": 0.35, "goals_scored": 1.1}
            wins  = sum(1 for _, r in games.iterrows() if r["away_score"] > r["home_score"])
            goals = sum(r["away_score"] for _, r in games.iterrows())
            return {"win_rate": wins/len(games), "goals_scored": goals/len(games)}

        def h2h_stats(home_t, away_t, n):
            h2h = df_past[
                ((df_past["home_team"] == home_t) & (df_past["away_team"] == away_t)) |
                ((df_past["home_team"] == away_t) & (df_past["away_team"] == home_t))
            ].tail(n)
            if len(h2h) == 0:
                return {"home_win_rate": 0.45, "avg_goals": 2.5, "btts_rate": 0.5, "n": 0}
            hw = btts = tg = 0
            for _, r in h2h.iterrows():
                hg = r["home_score"] if r["home_team"] == home_t else r["away_score"]
                ag = r["away_score"] if r["home_team"] == home_t else r["home_score"]
                tg += hg + ag
                if hg > ag: hw += 1
                if hg > 0 and ag > 0: btts += 1
            n_g = len(h2h)
            return {"home_win_rate": hw/n_g, "avg_goals": tg/n_g, "btts_rate": btts/n_g, "n": n_g}

        h5  = team_stats(home, 5);  h10 = team_stats(home, 10)
        hh5 = home_stats(home, 5)
        a5  = team_stats(away, 5);  a10 = team_stats(away, 10)
        aa5 = away_stats(away, 5)
        h2h = h2h_stats(home, away, 10)

        return [
            h5["win_rate"], h10["win_rate"], h5["draw_rate"],
            h5["goals_scored"], h5["goals_conceded"],
            hh5["win_rate"], hh5["goals_scored"],
            a5["win_rate"], a10["win_rate"], a5["draw_rate"],
            a5["goals_scored"], a5["goals_conceded"],
            aa5["win_rate"], aa5["goals_scored"],
            h2h["home_win_rate"], h2h["avg_goals"], h2h["btts_rate"],
            min(h2h["n"], 10) / 10,
            league_tier / 2,
        ]

    def _predict_with_api_models(self, features: list, home: str, away: str) -> Optional[Dict]:
        """Run full ensemble (XGB/LGBM/CatBoost/RF/ET/NN + ELO + Dixon-Coles) and combine via weighted voting."""
        if not self._api_models or features is None:
            return None

        X = np.array(features).reshape(1, -1)
        result_classes = {0: f"{away} to Win", 1: "Draw", 2: f"{home} to Win"}
        result_keys = {0: "away_win", 1: "draw", 2: "home_win"}
        preds = {}
        bma_inputs = []

        for model_name, model_info in self._api_models.items():
            try:
                model_obj = model_info["model"]
                is_nn = model_info["is_nn"]
                scaler = model_info["scaler"]

                # Neural networks need scaled features
                if is_nn and scaler is not None:
                    X_input = scaler.transform(X)
                else:
                    X_input = X

                proba = model_obj.predict_proba(X_input)[0]
                pred_idx = int(np.argmax(proba))

                # Use accuracy-based weight if available, else default weight
                accuracy_weight = self._api_model_weights.get(model_name, 0.5)
                # Scale confidence by model accuracy so better models count more
                raw_conf = float(proba[pred_idx]) * 100
                conf = raw_conf * accuracy_weight  # weighted confidence for voting

                if "match_result" in model_name:
                    label = result_classes.get(pred_idx, "Unknown")
                    pred_key = result_keys.get(pred_idx, "home_win")
                    prob_dict = {}
                    for ci, ck in result_keys.items():
                        prob_dict[ck] = float(proba[ci]) if ci < len(proba) else 0.0
                    bma_inputs.append({
                        "model_name": model_name,
                        "prediction": pred_key,
                        "confidence": float(proba[pred_idx]),
                        "probabilities": prob_dict,
                        "weight": accuracy_weight,
                    })
                elif "over_1_5" in model_name:
                    label = "Over 1.5 Goals" if pred_idx == 1 else "Under 1.5 Goals"
                elif "over_2_5" in model_name:
                    label = "Over 2.5 Goals" if pred_idx == 1 else "Under 2.5 Goals"
                elif "btts" in model_name:
                    label = "Both Teams to Score" if pred_idx == 1 else "Clean Sheet"
                else:
                    label = str(pred_idx)

                preds[model_name] = {
                    "prediction": label,
                    "confidence": round(raw_conf, 1),
                    "weighted_confidence": round(conf, 1),
                    "model_type": model_name.split("_")[-1],
                    "accuracy_weight": round(accuracy_weight, 4),
                }
            except Exception as e:
                logger.debug(f"Model {model_name} prediction failed: {e}")

        # Add ELO prediction
        elo_gap = 0.0
        if ELO_AVAILABLE and _elo_system is not None:
            try:
                elo_pred = _elo_system.predict(home, away)
                elo_gap = elo_pred.get("elo_gap", 0.0)
                preds["elo_rating"] = {
                    "prediction": elo_pred["prediction"].replace("_", " ").title(),
                    "confidence": elo_pred["confidence"] * 100,
                    "model_type": "elo",
                }
                bma_inputs.append({
                    "model_name": "elo_rating",
                    "prediction": elo_pred["prediction"],
                    "confidence": elo_pred["confidence"],
                    "probabilities": elo_pred.get("probabilities", {}),
                })
            except Exception as e:
                logger.debug(f"ELO prediction failed: {e}")

        # Add Dixon-Coles prediction
        if DIXON_COLES_AVAILABLE and _dixon_coles is not None and _dixon_coles.is_trained:
            try:
                dc_pred = _dixon_coles.predict(home, away)
                preds["dixon_coles"] = {
                    "prediction": dc_pred["prediction"].replace("_", " ").title(),
                    "confidence": dc_pred["confidence"] * 100,
                    "model_type": "dixon_coles",
                }
                bma_inputs.append({
                    "model_name": "dixon_coles",
                    "prediction": dc_pred["prediction"],
                    "confidence": dc_pred["confidence"],
                    "probabilities": dc_pred.get("probabilities", {}),
                })
            except Exception as e:
                logger.debug(f"Dixon-Coles prediction failed: {e}")

        if not preds:
            return None

        # Combine match-result predictions via BMA if available
        model_disagreement = 0.0
        if BMA_AVAILABLE and _bma is not None and len(bma_inputs) >= 2:
            try:
                combined = _bma.combine(bma_inputs)
                model_disagreement = combined.get("model_disagreement", 0.0)
            except Exception:
                pass
        elif len(bma_inputs) >= 2:
            from collections import Counter
            pred_vals = [p["prediction"] for p in bma_inputs]
            majority = Counter(pred_vals).most_common(1)[0]
            model_disagreement = 1.0 - majority[1] / len(pred_vals)

        # --- Weighted ensemble voting across all models ---
        # Group by prediction label, weight votes by model accuracy
        from collections import defaultdict
        vote_scores = defaultdict(float)
        vote_counts = defaultdict(int)
        raw_confidences = []

        for model_name, pred_info in preds.items():
            label = pred_info["prediction"]
            weight = pred_info.get("accuracy_weight", 0.5)
            raw_conf = pred_info.get("confidence", pred_info.get("weighted_confidence", 50.0))
            vote_scores[label] += weight * (raw_conf / 100.0)
            vote_counts[label] += 1
            raw_confidences.append(raw_conf)

        # Pick the prediction with the highest weighted score
        best_label = max(vote_scores, key=vote_scores.get)
        # Average raw confidence of models that voted for the winner
        winner_confs = [
            p["confidence"] for p in preds.values()
            if p["prediction"] == best_label
        ]
        ensemble_confidence = np.mean(winner_confs) if winner_confs else 50.0

        # Boost confidence when many models agree, penalize when they disagree
        agreement_ratio = vote_counts[best_label] / len(preds)
        # Scale: 1.0 = all agree, 0.5 = half agree
        confidence_adj = 0.8 + 0.2 * agreement_ratio  # 0.9 to 1.0 multiplier
        final_confidence = min(ensemble_confidence * confidence_adj, 95.0)

        return {
            "prediction": best_label,
            "confidence": round(final_confidence, 1),
            "model_predictions": preds,
            "model_disagreement": round(model_disagreement, 4),
            "elo_gap": round(elo_gap, 1),
            "ensemble_stats": {
                "models_voted": len(preds),
                "agreement_ratio": round(agreement_ratio, 2),
                "vote_scores": dict(vote_scores),
            },
        }

    def _resolve_bet_label(self, raw_label: str, model_predictions: Dict, home_team: str, away_team: str) -> str:
        """Convert raw model class labels to human-readable bet descriptions."""
        _LABEL_MAP = {
            "Home Win": f"{home_team} to Win",
            "Away Win": f"{away_team} to Win",
            "Draw": "Draw",
            "Over 2.5": "Over 2.5 Goals",
            "Under 2.5": "Under 2.5 Goals",
            "BTTS Yes": "Both Teams to Score",
            "BTTS No": "Clean Sheet",
            "Over 1.5": "Over 1.5 Goals",
            "Over 3.5": "Over 3.5 Goals",
        }
        if raw_label in _LABEL_MAP:
            return _LABEL_MAP[raw_label]

        # "Class 0/1" fallback — pick the most common readable prediction from sub-models
        readable = [
            v["prediction"] for v in model_predictions.values()
            if v.get("prediction") and not v["prediction"].startswith("Class")
        ]
        if readable:
            from collections import Counter
            return _LABEL_MAP.get(Counter(readable).most_common(1)[0][0],
                                  Counter(readable).most_common(1)[0][0])

        return "Value Bet"

    def _engineer_match_features_advanced(self, home_team: str, away_team: str, league: str) -> List[float]:
        """Engineer advanced features for a match using ML feature engineering."""
        try:
            if self.feature_engineer and ADVANCED_ML_AVAILABLE:
                # Normalise API team names to dataset naming convention
                norm_home = home_team
                norm_away = away_team
                if hasattr(self, "_historical_data_service"):
                    norm_home = self._historical_data_service.normalize_team_name(home_team)
                    norm_away = self._historical_data_service.normalize_team_name(away_team)

                features_df = self.feature_engineer.engineer_features_for_match(
                    norm_home, norm_away, league, datetime.now()
                )
                if features_df is not None and not features_df.empty:
                    return features_df.values.flatten().tolist()
                logger.warning(f"Feature engineering returned empty DataFrame for {home_team} vs {away_team}")

        except Exception as e:
            logger.error(f"❌ Feature engineering error: {str(e)}")

        return self._engineer_basic_features(home_team, away_team, league)

    def _engineer_basic_features(self, home_team: str, away_team: str, league: str) -> List[float]:
        """Basic feature engineering as fallback."""
        # Simple team strength mapping
        team_strengths = {
            "Arsenal": 0.8, "Chelsea": 0.8, "Manchester City": 0.9, "Liverpool": 0.85,
            "Manchester United": 0.75, "Tottenham": 0.7, "Real Madrid": 0.9, "Barcelona": 0.85,
            "Bayern Munich": 0.9, "PSG": 0.85, "Juventus": 0.8, "AC Milan": 0.75
        }

        home_strength = team_strengths.get(home_team, 0.5) + 0.1  # Home advantage
        away_strength = team_strengths.get(away_team, 0.5)

        # Create basic feature vector
        features = [
            home_strength,
            away_strength,
            home_strength - away_strength,  # Strength difference
            (home_strength + away_strength) / 2,  # Average strength
            1.0 if "Premier League" in league else 0.5,  # League strength
            0.1,  # Home advantage
        ]

        return features

    def _get_ensemble_predictions(self, features: List[float], home_team: str, away_team: str) -> Dict[str, Any]:
        """Get predictions from all available models."""
        predictions = {}

        # Convert features to numpy array for model prediction
        features_array = np.array(features).reshape(1, -1)

        for model_name, model_info in self.models.items():
            try:
                model = model_info["model"]

                # Get prediction based on model type
                if hasattr(model, 'predict_proba'):
                    # Classification model
                    probabilities = model.predict_proba(features_array)[0]
                    prediction_class = model.predict(features_array)[0]

                    # Map prediction to readable format
                    if "match_result" in model_name:
                        classes = ["Away Win", "Draw", "Home Win"]
                        prediction = classes[prediction_class] if prediction_class < len(classes) else "Home Win"
                    elif "over_under" in model_name:
                        prediction = "Over 2.5" if prediction_class == 1 else "Under 2.5"
                    elif "btts" in model_name:
                        prediction = "BTTS Yes" if prediction_class == 1 else "BTTS No"
                    else:
                        prediction = f"Class {prediction_class}"

                    confidence = max(probabilities) * 100

                else:
                    # Regression model
                    prediction_value = model.predict(features_array)[0]
                    prediction = f"Score: {prediction_value:.1f}"
                    confidence = 70.0  # Default confidence for regression

                predictions[model_name] = {
                    "prediction": prediction,
                    "confidence": round(confidence, 1),
                    "model_type": model_info["type"]
                }

            except Exception as e:
                logger.error(f"❌ Error with model {model_name}: {str(e)}")
                continue

        return predictions

    def _apply_meta_stacking(self, model_predictions: Dict[str, Any], features: List[float]) -> Dict[str, Any]:
        """Apply meta-model stacking to combine predictions."""
        if not model_predictions:
            return {"prediction": "No prediction", "confidence": 0.0}

        # Simple voting ensemble as meta-stacking
        prediction_votes = {}
        confidence_scores = []

        for model_name, pred_info in model_predictions.items():
            prediction = pred_info["prediction"]
            confidence = pred_info["confidence"]

            # Weight by model type (XGBoost gets higher weight)
            weight = 2.0 if "xgboost" in model_name else 1.5 if "enhanced" in model_name else 1.0

            if prediction not in prediction_votes:
                prediction_votes[prediction] = 0
            prediction_votes[prediction] += weight * (confidence / 100)
            confidence_scores.append(confidence)

        # Get the prediction with highest weighted vote
        if prediction_votes:
            final_prediction = max(prediction_votes.items(), key=lambda x: x[1])[0]
            meta_confidence = np.mean(confidence_scores)
        else:
            final_prediction = "No prediction"
            meta_confidence = 0.0

        return {
            "prediction": final_prediction,
            "confidence": round(meta_confidence, 1),
            "voting_scores": prediction_votes
        }

    def _generate_explanations(self, features: List[float], prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP/LIME explanations for the prediction."""
        explanations = {"available": False}

        if not self.explainers:
            return explanations

        try:
            # Use SHAP explainer if available
            for model_name, explainer_info in self.explainers.items():
                if explainer_info["type"] == "shap":
                    explainer = explainer_info["explainer"]

                    # Generate SHAP values
                    features_array = np.array(features).reshape(1, -1)
                    shap_values = explainer.shap_values(features_array)

                    explanations = {
                        "available": True,
                        "type": "shap",
                        "model": model_name,
                        "feature_importance": shap_values[0].tolist() if isinstance(shap_values, list) else shap_values.tolist(),
                        "explanation": "Feature importance scores from SHAP analysis"
                    }
                    break

        except Exception as e:
            logger.error(f"❌ Error generating explanations: {str(e)}")

        return explanations

    def _calculate_advanced_confidence(self, model_predictions: Dict[str, Any], final_prediction: Dict[str, Any]) -> float:
        """Calculate advanced confidence score based on model agreement."""
        if not model_predictions:
            return 0.0

        # Get confidence scores from all models
        confidences = [pred["confidence"] for pred in model_predictions.values()]

        # Calculate agreement score (how many models agree with final prediction)
        agreement_count = 0
        total_models = len(model_predictions)

        for pred_info in model_predictions.values():
            if pred_info["prediction"] == final_prediction["prediction"]:
                agreement_count += 1

        agreement_ratio = agreement_count / total_models if total_models > 0 else 0

        # Combine average confidence with agreement ratio
        avg_confidence = np.mean(confidences) if confidences else 0
        advanced_confidence = (avg_confidence * 0.7) + (agreement_ratio * 100 * 0.3)

        return round(min(advanced_confidence, 95.0), 1)  # Cap at 95%

    def _calculate_odds_from_confidence(self, confidence: float) -> float:
        """Calculate betting odds from confidence score."""
        if confidence >= 80:
            return round(1.2 + (100 - confidence) * 0.02, 2)
        elif confidence >= 60:
            return round(1.5 + (80 - confidence) * 0.05, 2)
        elif confidence >= 40:
            return round(2.0 + (60 - confidence) * 0.1, 2)
        else:
            return round(4.0 + (40 - confidence) * 0.2, 2)

    def _determine_odds_category(self, odds: float) -> str:
        """Determine odds category based on calculated odds."""
        if odds <= 2.5:
            return "2_odds"
        elif odds <= 5.0:
            return "5_odds"
        elif odds <= 10.0:
            return "10_odds"
        else:
            return "rollover"

    def _categorize_predictions_advanced(self, predictions: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize predictions using advanced logic."""
        categories = {
            "2_odds": [],
            "5_odds": [],
            "10_odds": [],
            "rollover": []
        }

        # Sort predictions by confidence (highest first)
        sorted_predictions = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)

        for prediction in sorted_predictions:
            category = prediction.get("category", "rollover")

            # Limit predictions per category for quality
            if len(categories[category]) < 5:  # Max 5 per category
                categories[category].append(prediction)
            elif category != "rollover" and len(categories["rollover"]) < 10:
                # Add high-quality predictions to rollover if category is full
                if prediction.get("confidence", 0) >= 70:
                    categories["rollover"].append(prediction)

        return categories

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        self._ensure_loaded()
        return {
            "total_models": len(self.models),
            "model_types": {
                "xgboost": len([m for m in self.models.keys() if "xgboost" in m]),
                "enhanced": len([m for m in self.models.keys() if "enhanced" in m]),
                "advanced": len([m for m in self.models.keys() if "advanced" in m]),
                "quick": len([m for m in self.models.keys() if "quick" in m])
            },
            "explainers": len(self.explainers),
            "advanced_features": {
                "feature_engineering": ADVANCED_ML_AVAILABLE,
                "shap_explanations": SHAP_AVAILABLE,
                "lime_explanations": LIME_AVAILABLE,
                "meta_stacking": True,
                "ensemble_voting": True
            },
            "models": list(self.models.keys())
        }

    def get_enhanced_predictions_with_explanations(self, date_str: Optional[str] = None,
                                                 include_explanations: bool = True,
                                                 explanation_detail: str = "human") -> Dict[str, Any]:
        """Get enhanced predictions with detailed explanations."""
        # Get base predictions
        result = self.get_predictions_for_date(date_str)

        if include_explanations and result["status"] == "success":
            # Add detailed explanations to each prediction
            for prediction in result["predictions"]:
                if "explanations" in prediction and prediction["explanations"]["available"]:
                    # Enhance explanations based on detail level
                    if explanation_detail == "human":
                        prediction["human_explanation"] = self._generate_human_explanation(prediction)
                    elif explanation_detail == "technical":
                        prediction["technical_explanation"] = self._generate_technical_explanation(prediction)
                    elif explanation_detail == "both":
                        prediction["human_explanation"] = self._generate_human_explanation(prediction)
                        prediction["technical_explanation"] = self._generate_technical_explanation(prediction)

        # Add enhanced metadata
        result["enhanced_features"] = {
            "explanations_included": include_explanations,
            "explanation_detail": explanation_detail,
            "model_info": self.get_model_info()
        }

        return result

    def _generate_human_explanation(self, prediction: Dict[str, Any]) -> str:
        """Generate human-readable explanation."""
        confidence = prediction.get("confidence", 0)
        home_team = prediction.get("home_team", "Home")
        away_team = prediction.get("away_team", "Away")
        pred_text = prediction.get("prediction", "Unknown")

        if confidence >= 80:
            certainty = "very confident"
        elif confidence >= 60:
            certainty = "confident"
        else:
            certainty = "moderately confident"

        return f"Our advanced ML models are {certainty} that {pred_text} in the {home_team} vs {away_team} match. This prediction is based on analysis of team form, historical performance, and advanced statistical modeling."

    def _generate_technical_explanation(self, prediction: Dict[str, Any]) -> str:
        """Generate technical explanation."""
        models_used = prediction.get("advanced_features", {}).get("models_used", 0)
        confidence = prediction.get("confidence", 0)

        return f"Prediction generated using ensemble of {models_used} ML models including XGBoost and enhanced algorithms. Meta-stacking applied with confidence score of {confidence}%. Feature engineering includes team strength, form analysis, and historical matchup data."


# Create global instance
advanced_prediction_service = AdvancedPredictionService()
