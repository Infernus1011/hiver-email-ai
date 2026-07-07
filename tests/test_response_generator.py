import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.response_generator import ResponseGenerator


class ResponseGeneratorTests(unittest.TestCase):
    def test_loads_few_shot_examples_from_dataset(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write('{"incoming": "Hi, I need a refund.", "expected_reply": "Hi, I can help with that."}\n')
            handle.write('{"incoming": "My app keeps crashing.", "expected_reply": "Sorry about that."}\n')
            dataset_path = handle.name

        try:
            generator = ResponseGenerator(preferred_provider="mock", dataset_path=dataset_path)
            self.assertEqual(len(generator.few_shot_examples), 2)
            self.assertIn("refund", generator.few_shot_examples[0]["incoming"].lower())
        finally:
            os.remove(dataset_path)


if __name__ == "__main__":
    unittest.main()
