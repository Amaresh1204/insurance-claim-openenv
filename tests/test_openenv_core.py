import unittest

from openenv_core import InsuranceClaimOpenEnv, StepAction


class OpenEnvCoreTests(unittest.TestCase):
    def setUp(self):
        self.env = InsuranceClaimOpenEnv(max_steps=3)
        self.env.reset(clear_json=True)

    def test_state_after_reset_is_clean(self):
        state = self.env.state()
        self.assertEqual(state.step_count, 0)
        self.assertEqual(state.total_reward, 0.0)
        self.assertFalse(state.episode_done)

    def test_step_progression_and_done_boundary(self):
        a1 = StepAction(pdf_path="sample.pdf", task_id="easy_fraud_detection", use_mock_model=True)
        a2 = StepAction(pdf_path="sample_valid.pdf", task_id="medium_policy_alignment", use_mock_model=True)
        a3 = StepAction(pdf_path="sample_valid.pdf", task_id="hard_calibrated_reasoning", use_mock_model=True)

        r1 = self.env.step(a1)
        self.assertFalse(r1.done)
        r2 = self.env.step(a2)
        self.assertFalse(r2.done)
        r3 = self.env.step(a3)
        self.assertTrue(r3.done)

        with self.assertRaises(ValueError):
            self.env.step(a1)

    def test_rewards_are_bounded(self):
        action = StepAction(pdf_path="sample_valid.pdf", task_id="hard_calibrated_reasoning", use_mock_model=True)
        result = self.env.step(action)
        self.assertGreaterEqual(result.reward, 0.0)
        self.assertLessEqual(result.reward, 1.0)


if __name__ == "__main__":
    unittest.main()
