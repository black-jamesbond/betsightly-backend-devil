#!/usr/bin/env python3
"""
Basketball Model Training Pipeline

Downloads real NBA data and trains basketball prediction models.
Uses free data sources and APIs.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from basketball.models import BasketballModelFactory
from basketball.feature_engineering import BasketballFeatureEngineer
from utils.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nba_data():
    """Download real NBA data from free sources."""
    print("🏀 DOWNLOADING REAL NBA DATA")
    print("=" * 40)
    
    try:
        # Try to use nba_api for real data
        from nba_api.stats.endpoints import leaguegamefinder
        from nba_api.stats.static import teams
        
        print("✅ NBA API available - downloading real data...")
        
        # Get all NBA teams
        nba_teams = teams.get_teams()
        print(f"📊 Found {len(nba_teams)} NBA teams")
        
        # Download games from recent seasons
        all_games = []
        seasons = ["2022-23", "2023-24"]  # Last 2 seasons
        
        for season in seasons:
            print(f"📅 Downloading {season} season data...")
            
            try:
                # Get games for this season
                game_finder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable="Regular Season"
                )
                
                games_df = game_finder.get_data_frames()[0]
                
                if not games_df.empty:
                    # Process the data
                    games_df['SEASON'] = season
                    all_games.append(games_df)
                    print(f"  ✅ Downloaded {len(games_df)} games")
                    
                    # Rate limit
                    import time
                    time.sleep(1)
                else:
                    print(f"  ⚠️  No games found for {season}")
                    
            except Exception as e:
                print(f"  ❌ Error downloading {season}: {str(e)}")
                continue
        
        if all_games:
            # Combine all seasons
            combined_df = pd.concat(all_games, ignore_index=True)
            
            # Save raw data
            data_dir = Path("data/basketball")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            raw_file = data_dir / "nba_games_raw.csv"
            combined_df.to_csv(raw_file, index=False)
            
            print(f"✅ Saved {len(combined_df)} total games to {raw_file}")
            return str(raw_file)
        else:
            print("❌ No real data downloaded, using fallback")
            return create_synthetic_data()
            
    except ImportError:
        print("⚠️  NBA API not available, installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nba_api"])
            print("✅ NBA API installed, please run again")
            return None
        except Exception as e:
            print(f"❌ Failed to install NBA API: {str(e)}")
            return create_synthetic_data()
    except Exception as e:
        print(f"❌ Error downloading real data: {str(e)}")
        return create_synthetic_data()

def create_synthetic_data():
    """Create synthetic NBA data for training if real data unavailable."""
    print("🎲 Creating synthetic NBA training data...")
    
    # NBA team names for realistic data
    teams = [
        "Los Angeles Lakers", "Boston Celtics", "Golden State Warriors",
        "Miami Heat", "Chicago Bulls", "San Antonio Spurs",
        "Philadelphia 76ers", "Denver Nuggets", "Milwaukee Bucks",
        "Phoenix Suns", "Brooklyn Nets", "Dallas Mavericks",
        "Toronto Raptors", "Atlanta Hawks", "Utah Jazz",
        "Portland Trail Blazers", "New York Knicks", "Cleveland Cavaliers",
        "Los Angeles Clippers", "Memphis Grizzlies", "New Orleans Pelicans",
        "Sacramento Kings", "Orlando Magic", "Minnesota Timberwolves",
        "Indiana Pacers", "Washington Wizards", "Charlotte Hornets",
        "Detroit Pistons", "Oklahoma City Thunder", "Houston Rockets"
    ]
    
    # Generate realistic game data
    np.random.seed(42)
    num_games = 2000  # About 2 seasons worth
    
    games_data = []
    
    for i in range(num_games):
        # Random teams
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        
        # Realistic NBA stats
        home_strength = np.random.uniform(0.3, 0.7)
        away_strength = np.random.uniform(0.3, 0.7)
        
        # Home court advantage
        home_advantage = 0.1
        home_win_prob = home_strength + home_advantage - away_strength
        home_win_prob = np.clip(home_win_prob, 0.1, 0.9)
        
        home_wins = np.random.random() < home_win_prob
        
        # Generate realistic scores
        base_pace = np.random.uniform(95, 105)  # Possessions per game
        home_efficiency = np.random.uniform(1.0, 1.2)  # Points per possession
        away_efficiency = np.random.uniform(1.0, 1.2)
        
        home_score = int(base_pace * home_efficiency + np.random.normal(0, 5))
        away_score = int(base_pace * away_efficiency + np.random.normal(0, 5))
        
        # Ensure winner has higher score
        if home_wins and home_score <= away_score:
            home_score = away_score + np.random.randint(1, 10)
        elif not home_wins and away_score <= home_score:
            away_score = home_score + np.random.randint(1, 10)
        
        # Realistic ranges
        home_score = np.clip(home_score, 85, 140)
        away_score = np.clip(away_score, 85, 140)
        
        game_data = {
            'GAME_ID': f'synthetic_{i:06d}',
            'GAME_DATE': (datetime.now() - timedelta(days=np.random.randint(0, 730))).strftime('%Y-%m-%d'),
            'HOME_TEAM': home_team,
            'AWAY_TEAM': away_team,
            'HOME_PTS': home_score,
            'AWAY_PTS': away_score,
            'TOTAL_PTS': home_score + away_score,
            'HOME_WIN': 1 if home_wins else 0,
            'HOME_FG_PCT': np.random.uniform(0.40, 0.55),
            'AWAY_FG_PCT': np.random.uniform(0.40, 0.55),
            'HOME_FG3_PCT': np.random.uniform(0.30, 0.45),
            'AWAY_FG3_PCT': np.random.uniform(0.30, 0.45),
            'HOME_FT_PCT': np.random.uniform(0.70, 0.85),
            'AWAY_FT_PCT': np.random.uniform(0.70, 0.85),
            'HOME_REB': np.random.randint(40, 55),
            'AWAY_REB': np.random.randint(40, 55),
            'HOME_AST': np.random.randint(20, 35),
            'AWAY_AST': np.random.randint(20, 35),
            'HOME_TOV': np.random.randint(10, 20),
            'AWAY_TOV': np.random.randint(10, 20),
            'SEASON': np.random.choice(['2022-23', '2023-24'])
        }
        
        games_data.append(game_data)
    
    # Create DataFrame
    df = pd.DataFrame(games_data)
    
    # Save synthetic data
    data_dir = Path("data/basketball")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    synthetic_file = data_dir / "nba_games_synthetic.csv"
    df.to_csv(synthetic_file, index=False)
    
    print(f"✅ Created {len(df)} synthetic games")
    print(f"📊 Average total points: {df['TOTAL_PTS'].mean():.1f}")
    print(f"📊 Home win rate: {df['HOME_WIN'].mean():.1%}")
    print(f"💾 Saved to {synthetic_file}")
    
    return str(synthetic_file)

def transform_nba_data_to_games(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transform NBA API team-game data to game-level data."""
    print("🔄 Processing NBA team-game records...")

    # Group by GAME_ID to get both teams for each game
    games_list = []

    for game_id, game_group in raw_df.groupby('GAME_ID'):
        if len(game_group) != 2:
            continue  # Skip games without exactly 2 teams

        # Sort by MATCHUP to identify home/away (home team has 'vs', away has '@')
        game_group = game_group.sort_values('MATCHUP')

        # Determine home and away teams
        team1 = game_group.iloc[0]
        team2 = game_group.iloc[1]

        # Home team typically has 'vs' in matchup, away has '@'
        if 'vs.' in team1['MATCHUP']:
            home_team_data = team1
            away_team_data = team2
        elif '@' in team1['MATCHUP']:
            home_team_data = team2
            away_team_data = team1
        else:
            # Fallback: first team is home
            home_team_data = team1
            away_team_data = team2

        # Create game record
        game_record = {
            'GAME_ID': game_id,
            'GAME_DATE': home_team_data['GAME_DATE'],
            'SEASON': home_team_data['SEASON'],
            'HOME_TEAM': home_team_data['TEAM_NAME'],
            'AWAY_TEAM': away_team_data['TEAM_NAME'],
            'HOME_PTS': home_team_data['PTS'],
            'AWAY_PTS': away_team_data['PTS'],
            'TOTAL_PTS': home_team_data['PTS'] + away_team_data['PTS'],
            'HOME_WIN': 1 if home_team_data['WL'] == 'W' else 0,
            'HOME_FG_PCT': home_team_data['FG_PCT'],
            'AWAY_FG_PCT': away_team_data['FG_PCT'],
            'HOME_FG3_PCT': home_team_data['FG3_PCT'],
            'AWAY_FG3_PCT': away_team_data['FG3_PCT'],
            'HOME_FT_PCT': home_team_data['FT_PCT'],
            'AWAY_FT_PCT': away_team_data['FT_PCT'],
            'HOME_REB': home_team_data['REB'],
            'AWAY_REB': away_team_data['REB'],
            'HOME_AST': home_team_data['AST'],
            'AWAY_AST': away_team_data['AST'],
            'HOME_TOV': home_team_data['TOV'],
            'AWAY_TOV': away_team_data['TOV'],
            'HOME_PLUS_MINUS': home_team_data['PLUS_MINUS'],
            'AWAY_PLUS_MINUS': away_team_data['PLUS_MINUS']
        }

        games_list.append(game_record)

    # Create DataFrame
    games_df = pd.DataFrame(games_list)

    # Convert date column
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])

    print(f"✅ Transformed {len(games_df)} games from {len(raw_df)} team records")
    print(f"📊 Date range: {games_df['GAME_DATE'].min()} to {games_df['GAME_DATE'].max()}")
    print(f"📊 Average total points: {games_df['TOTAL_PTS'].mean():.1f}")
    print(f"📊 Home win rate: {games_df['HOME_WIN'].mean():.1%}")

    return games_df

def train_basketball_models(data_file: str):
    """Train basketball prediction models."""
    print("\n🤖 TRAINING BASKETBALL MODELS")
    print("=" * 40)

    try:
        # Load data
        print(f"📂 Loading data from {data_file}")
        raw_df = pd.read_csv(data_file)
        print(f"📊 Loaded {len(raw_df)} team-game records")

        # Transform NBA API data to game format
        print("🔄 Transforming NBA data to game format...")
        df = transform_nba_data_to_games(raw_df)
        print(f"📊 Created {len(df)} game records")

        if df.empty:
            print("❌ No games created from NBA data")
            return False

        # Initialize components
        model_factory = BasketballModelFactory()
        feature_engineer = BasketballFeatureEngineer()

        # Engineer features
        print("🔧 Engineering features...")
        features_df = feature_engineer.engineer_features(df)
        
        if features_df.empty:
            print("❌ No features generated")
            return False
        
        print(f"✅ Generated {len(features_df.columns)} features")
        
        # Prepare training data
        feature_columns = feature_engineer.get_feature_names()
        X = features_df[feature_columns]
        
        # Train Win/Loss model
        print("\n🎯 Training Win/Loss Model (XGBoost)...")
        win_loss_model = model_factory.create_win_loss_model()
        y_win_loss = df['HOME_WIN']
        
        wl_results = win_loss_model.train(X, y_win_loss)
        print(f"✅ Win/Loss Model - Accuracy: {wl_results['accuracy']:.3f}")
        
        # Train Over/Under model
        print("\n📊 Training Over/Under Model (LightGBM)...")
        over_under_model = model_factory.create_over_under_model()
        y_total = df['TOTAL_PTS']
        
        ou_results = over_under_model.train(X, y_total, total_threshold=220.0)
        print(f"✅ Over/Under Model - Accuracy: {ou_results['accuracy']:.3f}")
        
        # Train Neural Network model
        print("\n🧠 Training Neural Network Model...")
        nn_model = model_factory.create_neural_network_model()
        if nn_model:
            nn_results = nn_model.train(X, y_win_loss, 'classification')
            print(f"✅ Neural Network Model - Accuracy: {nn_results['accuracy']:.3f}")
        else:
            print("⚠️  Neural Network not available")
            nn_results = {}
        
        # Summary
        print("\n🎉 TRAINING COMPLETE!")
        print("=" * 40)
        print(f"✅ Win/Loss (XGBoost): {wl_results['accuracy']:.1%} accuracy")
        print(f"✅ Over/Under (LightGBM): {ou_results['accuracy']:.1%} accuracy")
        if nn_results:
            print(f"✅ Neural Network: {nn_results['accuracy']:.1%} accuracy")
        
        print(f"\n📁 Models saved to: basketball/models/")
        print("🚀 Basketball prediction system ready!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training pipeline."""
    print("🏀 NBA BASKETBALL MODEL TRAINING PIPELINE")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Download data
    data_file = download_nba_data()
    
    if not data_file:
        print("❌ Failed to get training data")
        return False
    
    # Step 2: Train models
    success = train_basketball_models(data_file)
    
    if success:
        print("\n🎉 BASKETBALL TRAINING PIPELINE COMPLETE!")
        print("✅ Models trained and ready for predictions")
        print("🔗 Test with: curl http://localhost:8000/api/basketball-predictions/")
    else:
        print("\n❌ Training pipeline failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
