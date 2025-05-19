import unittest
from huggingface_hub import InferenceClient

# Actual working values

class TestHuggingFaceEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = InferenceClient(model=ENDPOINT_URL, token=HF_TOKEN)

    def test_basic_generation(self):
        prompt = "The capital of France is"
        response = self.client.text_generation(prompt, max_new_tokens=10)
        print("\n[test_basic_generation]:", response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), len(prompt))

    def test_max_token_limit(self):
        prompt = "Repeat after me:"
        response = self.client.text_generation(prompt, max_new_tokens=2)
        print("\n[test_max_token_limit]:", response)
        self.assertLessEqual(len(response.split()), len(prompt.split()) + 3)

    def test_deterministic_output(self):
        prompt = "2 + 2 equals"
        response1 = self.client.text_generation(prompt, max_new_tokens=5, do_sample=False)
        response2 = self.client.text_generation(prompt, max_new_tokens=5, do_sample=False)
        print("\n[test_deterministic_output]:", response1)
        self.assertEqual(response1, response2)

    def test_empty_prompt(self):
        prompt = ""
        with self.assertRaises(Exception):
            self.client.text_generation(prompt, max_new_tokens=5)

    def test_multilingual(self):
        prompt = "Translate 'hello' to Spanish:"
        response = self.client.text_generation(prompt, max_new_tokens=10)
        print("\n[test_multilingual]:", response)
        self.assertTrue(any(word in response.lower() for word in ["hola", "saludo", "buenos"]))

if __name__ == "__main__":
    unittest.main()
