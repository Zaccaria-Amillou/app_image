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

        # Check that the status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Check that the content type is 'text/html; charset=utf-8'
        self.assertEqual(response.content_type, 'text/html; charset=utf-8')

        # Check that certain strings are in the response data
        self.assertIn(b'Flask App', response.data)
        self.assertIn(b'<!DOCTYPE html>', response.data)
        self.assertIn(b'<html lang="en">', response.data)
        self.assertIn(b'<body>', response.data)
        self.assertIn(b'</body>', response.data)
        self.assertIn(b'</html>', response.data)

if __name__ == "__main__":
    unittest.main()
