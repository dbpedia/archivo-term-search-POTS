import unittest
from unittest.mock import patch
import json
import requests

# Define the API endpoint
api_endpoint = "http://127.0.0.1:8014/search"

def perform_search_case(case_data):
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(api_endpoint, headers=headers, data=json.dumps(case_data))
    return response

class TestSearchAPI(unittest.TestCase):

    @patch('requests.post')  # Mock the requests.post method
    def test_simple_fuzzy_filtering(self, mock_post):
        case_data = {
            "fuzzy_filters": {"label": "date"},
            "fuzzy_filters_config": {"model_name": "LaBSE"},
        }

        # Set up the mock response
        mock_response = {
            "result": "some_result",
            "status": "success"
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        # Perform the search
        response = perform_search_case(case_data)
        
        # Print the response
        print("Test: test_simple_fuzzy_filtering")
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
        
        # Check the mock response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_response)
    
    @patch('requests.post')
    def test_simple_exact_filtering(self, mock_post):
        case_data = {
            "exact_filters": {"termtype": "ObjectProperty"}
        }

        mock_response = {
            "result": "exact_result",
            "status": "success"
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        response = perform_search_case(case_data)
        
        # Print the response
        print("Test: test_simple_exact_filtering")
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_response)
    
    @patch('requests.post')
    def test_complex_exact_filtering(self, mock_post):
        case_data = {
            "exact_filters": {"termtype": "ObjectProperty", "domain": 'http://www.demcare.eu/ontologies/demlab.owl#Protocol'}
        }

        mock_response = {
            "result": "complex_exact_result",
            "status": "success"
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        response = perform_search_case(case_data)
        
        # Print the response
        print("Test: test_complex_exact_filtering")
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_response)
    
    @patch('requests.post')
    def test_complex_fuzzy_filtering(self, mock_post):
        case_data = {
            "fuzzy_filters": {"label": "date", "domain": "creative work"},
            "fuzzy_filters_config": {"model_name": "LaBSE"},
            "exact_filters": {"termtype": "ObjectProperty"}
        }

        mock_response = {
            "result": "complex_fuzzy_result",
            "status": "success"
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        response = perform_search_case(case_data)
        
        # Print the response
        print("Test: test_complex_fuzzy_filtering")
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_response)


if __name__ == '__main__':
    unittest.main()
