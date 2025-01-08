import unittest
import os
import numpy as np
from recomendacao_improved import (
    AppConfig,
    ImageProcessor,
    ImageDatabase,
    ImageRecommender
)
from unittest.mock import patch, MagicMock

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.processor = ImageProcessor(self.config)
        
    @patch('cv2.imread')
    def test_preprocess_image(self, mock_imread):
        # Setup mock
        mock_imread.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Test valid image
        result = self.processor.preprocess_image('test.jpg')
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (1, 224, 224, 3))
        
        # Test invalid image
        mock_imread.return_value = None
        result = self.processor.preprocess_image('invalid.jpg')
        self.assertIsNone(result)

class TestImageDatabase(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.database = ImageDatabase(self.config)
        
    @patch('pandas.read_csv')
    def test_load_features(self, mock_read_csv):
        # Setup mock
        mock_df = MagicMock()
        mock_df.iloc = MagicMock()
        mock_df.iloc.__getitem__.return_value.values = np.zeros((10, 128))
        mock_df.__getitem__.return_value.values = ['img1', 'img2']
        mock_read_csv.return_value = mock_df
        
        # Test successful load
        self.database.load_features()
        self.assertIsNotNone(self.database.features)
        self.assertIsNotNone(self.database.labels)
        
        # Test file not found
        mock_read_csv.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.database.load_features()

class TestImageRecommender(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.recommender = ImageRecommender(self.config)
        
    @patch.object(ImageProcessor, 'preprocess_image')
    @patch.object(ImageProcessor, 'extract_features')
    @patch.object(ImageDatabase, 'find_similar_images')
    def test_process_uploaded_image(self, mock_find, mock_extract, mock_preprocess):
        # Setup mocks
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3))
        mock_extract.return_value = np.zeros((1, 128))
        mock_find.return_value = [('img1', 0.9), ('img2', 0.8)]
        
        # Test successful processing
        result = self.recommender.process_uploaded_image('test.jpg')
        self.assertEqual(len(result), 2)
        
        # Test invalid image
        mock_preprocess.return_value = None
        result = self.recommender.process_uploaded_image('invalid.jpg')
        self.assertEqual(len(result), 0)

if __name__ == '__main__':
    unittest.main()
