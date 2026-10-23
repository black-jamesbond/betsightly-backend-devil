"""Training results cannot be invented or silently promoted to production."""
import json
import tempfile
import unittest
from pathlib import Path

from services.training_pipeline_service import TrainingPipelineService


class TrainingValidationTests(unittest.TestCase):
    def setUp(self):
        self.service = TrainingPipelineService()
        self.tmp = tempfile.TemporaryDirectory()
        self.service._last_candidate_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_weights_report_no_accuracy(self):
        self.assertEqual(self.service._evaluate_new_models(), {})

    def test_measured_weights_drive_accuracy(self):
        weights = {
            "match_result_rf": 0.61, "match_result_xgb": 0.64,
            "over_1_5_rf": 0.75, "over_2_5_rf": 0.67,
            "btts_rf": 0.70,
        }
        (self.service._last_candidate_dir / "model_weights.json").write_text(
            json.dumps(weights))
        results = self.service._evaluate_new_models()
        self.assertEqual(results["match_result"], 0.64)
        self.assertEqual(results["over_1_5"], 0.75)
        self.assertAlmostEqual(results["overall"], (0.64 + .75 + .67 + .70) / 4)

    def test_partial_measured_results_cannot_be_called_complete(self):
        (self.service._last_candidate_dir / "model_weights.json").write_text(
            json.dumps({"match_result_rf": .66}))
        self.assertEqual(self.service._evaluate_new_models(), {})

    def test_automatic_model_promotion_is_disabled(self):
        self.assertFalse(self.service._should_deploy_models({"overall": .40}))
        self.assertFalse(self.service._deploy_new_models())

    def test_training_config_refers_to_existing_scripts(self):
        for path in self.service.training_scripts.values():
            self.assertTrue((self.service.root / path).exists(), path)


if __name__ == "__main__":
    unittest.main()
