import unittest
import json
import numpy as np
from app import app, get_clip_vector  # Import your Flask app and functions

class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()  # Create a test client for making HTTP requests
        cls.client.testing = True

    def test_index_route(self):
        # Test the index route
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)  # Check if HTML is returned

    def test_get_clip_vector_text(self):
        # Test get_clip_vector with text input
        sample_text = "A test sentence for CLIP."
        vector = get_clip_vector(sample_text, is_image=False)
        self.assertEqual(vector.shape, (512,))
        self.assertTrue(np.all(np.isfinite(vector)))  # Check for valid finite numbers

    def test_search_route(self):
        # Mock data for testing search route
        mock_data = {
            "queryText": "Test query",
            "top_k": 3
        }

        # Send a POST request to /search
        response = self.client.post('/search', data=json.dumps(mock_data),
                                    content_type='application/json')
        
        # Check if the response code is 200 OK
        self.assertEqual(response.status_code, 200)

        # Parse the JSON response
        response_data = response.get_json()
        self.assertIsInstance(response_data, list)  # Expect a list of results
        for item in response_data:
            self.assertIn("rank", item)
            self.assertIn("id", item)
            self.assertIn("score", item)

    def test_search_route_invalid_data(self):
        # Test the search route with missing data
        response = self.client.post('/search', data=json.dumps({}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 500)
        response_data = response.get_json()
        self.assertIn("message", response_data)
        self.assertIn("error", response_data)

if __name__ == '__main__':
    unittest.main()
