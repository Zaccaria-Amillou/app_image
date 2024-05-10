import unittest
from flask_app import app

class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True 

    def test_homepage_status_code(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_get_prediction_status_code(self):
        response = self.app.get('/prediction')
        self.assertEqual(response.status_code, 200)

    def test_homepage_content(self):
        response = self.app.get('/')
        self.assertIn(b'Flask App', response.data)

    def test_get_prediction_content(self):
        response = self.app.get('/prediction')
        self.assertIn(b'prediction.html', response.data)

if __name__ == "__main__":
    unittest.main()
