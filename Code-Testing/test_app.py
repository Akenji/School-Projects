import unittest
from app import app


class TestApp(unittest.TestCase): #defines a new class called TestApp
    def setUp(self): #start of the setup methode which is a special method in unittest that is run b4 any method i the class
        self.app = app.test_client()

    def test_home_route(self): #this method defines a test method that sarts with the word test
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"message": "Hello level 400 FET, Quality Assurance!"})


if __name__ == '_main_':
    unittest.main()