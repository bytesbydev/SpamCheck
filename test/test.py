import unittest
from src.preprocessing import preprocess_text


class TestPreprocessing(unittest.TestCase):

    def test_lowercase(self):
        result = preprocess_text("HELLO WORLD")
        self.assertEqual(result, result.lower())

    def test_remove_punctuation(self):
        result = preprocess_text("Hello, world!!!")
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)

    def test_remove_stopwords(self):
        result = preprocess_text("this is a simple test")
        self.assertNotIn("is", result.split())

    def test_remove_numbers(self):
        result = preprocess_text("Win 1000 dollars now")
        self.assertNotIn("1000", result)

    def test_empty_string(self):
        result = preprocess_text("")
        self.assertEqual(result, "")

    def test_real_spam_message(self):
        msg = "Congratulations! You won a FREE iPhone."
        result = preprocess_text(msg)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()