"""Add training pipeline and caching tables

Revision ID: add_training_cache_tables
Revises: 
Create Date: 2025-06-03 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_training_cache_tables'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Add training pipeline and caching tables."""
    
    # Create cached_predictions_v2 table
    op.create_table('cached_predictions_v2',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('prediction_date', sa.String(), nullable=False),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('home_team', sa.String(), nullable=False),
        sa.Column('away_team', sa.String(), nullable=False),
        sa.Column('league', sa.String(), nullable=False),
        sa.Column('fixture_id', sa.String(), nullable=True),
        sa.Column('match_time', sa.String(), nullable=True),
        sa.Column('prediction', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('odds', sa.Float(), nullable=False),
        sa.Column('models_used', sa.JSON(), nullable=True),
        sa.Column('service_used', sa.String(), nullable=True),
        sa.Column('model_predictions', sa.JSON(), nullable=True),
        sa.Column('explanations', sa.JSON(), nullable=True),
        sa.Column('advanced_features', sa.JSON(), nullable=True),
        sa.Column('generated_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('is_stale', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for cached_predictions_v2
    op.create_index('idx_cached_pred_date_category', 'cached_predictions_v2', ['prediction_date', 'category'])
    op.create_index('idx_cached_pred_generated', 'cached_predictions_v2', ['generated_at'])
    op.create_index('idx_cached_pred_expires', 'cached_predictions_v2', ['expires_at'])
    op.create_index(op.f('ix_cached_predictions_v2_category'), 'cached_predictions_v2', ['category'])
    op.create_index(op.f('ix_cached_predictions_v2_prediction_date'), 'cached_predictions_v2', ['prediction_date'])
    
    # Create prediction_batches table
    op.create_table('prediction_batches',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('batch_date', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('fixtures_fetched', sa.Integer(), nullable=True),
        sa.Column('predictions_generated', sa.Integer(), nullable=True),
        sa.Column('categories_populated', sa.JSON(), nullable=True),
        sa.Column('generation_time_seconds', sa.Float(), nullable=True),
        sa.Column('api_source', sa.String(), nullable=True),
        sa.Column('models_loaded', sa.Integer(), nullable=True),
        sa.Column('service_used', sa.String(), nullable=True),
        sa.Column('advanced_features_used', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('warnings', sa.JSON(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('batch_date')
    )
    
    # Create index for prediction_batches
    op.create_index(op.f('ix_prediction_batches_batch_date'), 'prediction_batches', ['batch_date'])
    
    # Create model_training_runs table
    op.create_table('model_training_runs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('training_date', sa.String(), nullable=False),
        sa.Column('training_type', sa.String(), nullable=False),
        sa.Column('trigger_reason', sa.String(), nullable=True),
        sa.Column('training_data_size', sa.Integer(), nullable=True),
        sa.Column('new_matches_added', sa.Integer(), nullable=True),
        sa.Column('data_source', sa.String(), nullable=True),
        sa.Column('models_trained', sa.JSON(), nullable=True),
        sa.Column('training_algorithms', sa.JSON(), nullable=True),
        sa.Column('training_time_seconds', sa.Float(), nullable=True),
        sa.Column('previous_accuracy', sa.JSON(), nullable=True),
        sa.Column('new_accuracy', sa.JSON(), nullable=True),
        sa.Column('performance_improvement', sa.JSON(), nullable=True),
        sa.Column('deployment_status', sa.String(), nullable=True),
        sa.Column('deployment_reason', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('warnings', sa.JSON(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('deployed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create index for model_training_runs
    op.create_index(op.f('ix_model_training_runs_training_date'), 'model_training_runs', ['training_date'])
    
    # Create prediction_accuracy table
    op.create_table('prediction_accuracy',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('match_date', sa.String(), nullable=False),
        sa.Column('home_team', sa.String(), nullable=False),
        sa.Column('away_team', sa.String(), nullable=False),
        sa.Column('league', sa.String(), nullable=False),
        sa.Column('prediction_type', sa.String(), nullable=False),
        sa.Column('predicted_outcome', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('odds', sa.Float(), nullable=False),
        sa.Column('actual_outcome', sa.String(), nullable=True),
        sa.Column('home_score', sa.Integer(), nullable=True),
        sa.Column('away_score', sa.Integer(), nullable=True),
        sa.Column('is_correct', sa.Boolean(), nullable=True),
        sa.Column('accuracy_score', sa.Float(), nullable=True),
        sa.Column('model_used', sa.String(), nullable=False),
        sa.Column('service_used', sa.String(), nullable=False),
        sa.Column('predicted_at', sa.DateTime(), nullable=False),
        sa.Column('result_recorded_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for prediction_accuracy
    op.create_index('idx_accuracy_date_model', 'prediction_accuracy', ['match_date', 'model_used'])
    op.create_index('idx_accuracy_prediction_type', 'prediction_accuracy', ['prediction_type'])
    op.create_index(op.f('ix_prediction_accuracy_match_date'), 'prediction_accuracy', ['match_date'])
    
    # Create cache_status table
    op.create_table('cache_status',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('cache_date', sa.String(), nullable=False),
        sa.Column('total_predictions', sa.Integer(), nullable=True),
        sa.Column('predictions_by_category', sa.JSON(), nullable=True),
        sa.Column('cache_hit_rate', sa.Float(), nullable=True),
        sa.Column('average_response_time_ms', sa.Float(), nullable=True),
        sa.Column('cache_size_mb', sa.Float(), nullable=True),
        sa.Column('last_refresh_at', sa.DateTime(), nullable=True),
        sa.Column('next_refresh_at', sa.DateTime(), nullable=True),
        sa.Column('is_stale', sa.Boolean(), nullable=True),
        sa.Column('health_status', sa.String(), nullable=True),
        sa.Column('health_issues', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('cache_date')
    )
    
    # Create index for cache_status
    op.create_index(op.f('ix_cache_status_cache_date'), 'cache_status', ['cache_date'])


def downgrade():
    """Remove training pipeline and caching tables."""
    
    # Drop indexes first
    op.drop_index(op.f('ix_cache_status_cache_date'), table_name='cache_status')
    op.drop_index('idx_accuracy_prediction_type', table_name='prediction_accuracy')
    op.drop_index('idx_accuracy_date_model', table_name='prediction_accuracy')
    op.drop_index(op.f('ix_prediction_accuracy_match_date'), table_name='prediction_accuracy')
    op.drop_index(op.f('ix_model_training_runs_training_date'), table_name='model_training_runs')
    op.drop_index(op.f('ix_prediction_batches_batch_date'), table_name='prediction_batches')
    op.drop_index('idx_cached_pred_expires', table_name='cached_predictions_v2')
    op.drop_index('idx_cached_pred_generated', table_name='cached_predictions_v2')
    op.drop_index('idx_cached_pred_date_category', table_name='cached_predictions_v2')
    op.drop_index(op.f('ix_cached_predictions_v2_prediction_date'), table_name='cached_predictions_v2')
    op.drop_index(op.f('ix_cached_predictions_v2_category'), table_name='cached_predictions_v2')
    
    # Drop tables
    op.drop_table('cache_status')
    op.drop_table('prediction_accuracy')
    op.drop_table('model_training_runs')
    op.drop_table('prediction_batches')
    op.drop_table('cached_predictions_v2')
