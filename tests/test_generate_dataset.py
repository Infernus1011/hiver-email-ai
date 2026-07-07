import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_dataset import generate_email_pair


class GenerateDatasetTests(unittest.TestCase):
    def test_generate_email_pair_does_not_raise_key_error(self):
        pair = generate_email_pair(0)

        self.assertEqual(pair.id, "email_00000")
        self.assertIn(pair.category, {"billing", "technical", "feature_request", "account", "general"})
        self.assertTrue(pair.incoming)
        self.assertTrue(pair.expected_reply)


if __name__ == "__main__":
    unittest.main()
