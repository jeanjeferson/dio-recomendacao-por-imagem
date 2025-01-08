import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import cv2
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import asyncio
from typing import Awaitable

# Configuration
@dataclass
class AppConfig:
    UPLOAD_FOLDER: str = 'static/uploads'
    DATABASE_FOLDER: str = 'static/database'
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS: set = {'png', 'jpg', 'jpeg'}
    FEATURES_FILE: str = 'image_features.csv'
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    MAX_WORKERS: int = 4  # Number of threads for CPU-bound tasks
    
    def validate(self):
        """Validate configuration values"""
        if not os.path.exists(self.DATABASE_FOLDER):
            raise ValueError(f"Database folder {self.DATABASE_FOLDER} does not exist")
            
        if not os.path.exists(self.UPLOAD_FOLDER):
            os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)

class ImageProcessor:
    """Handles image processing and feature extraction with async support"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = self._load_vgg16_model()
        self.logger = self._setup_logger()
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
    @lru_cache(maxsize=1)
    def _load_vgg16_model(self) -> Model:
        """Load and cache VGG16 model without classification layer"""
        try:
            base_model = VGG16(weights='imagenet', 
                             include_top=False, 
                             input_shape=(*self.config.IMAGE_SIZE, 3))
            return Model(inputs=base_model.input, 
                        outputs=base_model.output)
        except Exception as e:
            self.logger.error(f"Failed to load VGG16 model: {str(e)}")
            raise
            
    def _setup_logger(self) -> logging.Logger:
        """Configure and return logger"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    async def preprocess_image_async(self, img_path: str) -> Optional[np.ndarray]:
        """Async version of image preprocessing"""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self.executor,
                self.preprocess_image,
                img_path
            )
        except Exception as e:
            self.logger.error(f"Async preprocessing error: {str(e)}")
            return None
        
    def preprocess_image(self, img_path: str) -> Optional[np.ndarray]:
        """Preprocess image for VGG16 model"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to read image: {img_path}")
                
            img = cv2.resize(img, self.config.IMAGE_SIZE)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            return preprocess_input(img)
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            return None
            
    async def extract_features_async(self, img: np.ndarray) -> np.ndarray:
        """Async version of feature extraction"""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self.executor,
                self.extract_features,
                img
            )
        except Exception as e:
            self.logger.error(f"Async feature extraction error: {str(e)}")
            raise
            
    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """Extract features from preprocessed image"""
        try:
            features = self.model.predict(img)
            return features.reshape(features.shape[0], -1)
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            raise

class ImageDatabase:
    """Manages the image database and similarity search with async support"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.features = None
        self.labels = None
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
    async def load_features_async(self) -> None:
        """Async version of load_features"""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                self.executor,
                self.load_features
            )
        except Exception as e:
            self.logger.error(f"Async feature loading error: {str(e)}")
            raise
            
    def load_features(self) -> None:
        """Load features from CSV file"""
        try:
            if not os.path.exists(self.config.FEATURES_FILE):
                raise FileNotFoundError(
                    f"Features file {self.config.FEATURES_FILE} not found"
                )
                
            df = pd.read_csv(self.config.FEATURES_FILE)
            self.features = df.iloc[:, :-1].values
            self.labels = df['label'].values
            self.logger.info(f"Loaded {len(self.labels)} image features")
        except Exception as e:
            self.logger.error(f"Failed to load features: {str(e)}")
            raise
            
    async def find_similar_images_async(self, 
                                      query_features: np.ndarray, 
                                      top_n: int = 5) -> List[Tuple[str, float]]:
        """Async version of find_similar_images"""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self.executor,
                self.find_similar_images,
                query_features,
                top_n
            )
        except Exception as e:
            self.logger.error(f"Async similarity search error: {str(e)}")
            return []
            
    def find_similar_images(self, 
                          query_features: np.ndarray, 
                          top_n: int = 5) -> List[Tuple[str, float]]:
        """Find similar images using cosine similarity"""
        if self.features is None:
            raise ValueError("Features not loaded")
            
        try:
            similarities = cosine_similarity(query_features, self.features)
            indices = np.argsort(similarities[0])[::-1][1:top_n+1]
            
            return [(self.labels[i], float(similarities[0][i])) 
                    for i in indices]
        except Exception as e:
            self.logger.error(f"Error finding similar images: {str(e)}")
            return []

class ImageRecommender:
    """Main class for image recommendation system with async support"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.processor = ImageProcessor(config)
        self.database = ImageDatabase(config)
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        self.logger = logging.getLogger(__name__)
        
        # Validate database directory
        if not os.path.exists(config.DATABASE_FOLDER):
            os.makedirs(config.DATABASE_FOLDER)
            raise FileNotFoundError(
                f"Database directory {config.DATABASE_FOLDER} is empty"
            )
            
        # Check for images
        image_files = [f for f in os.listdir(config.DATABASE_FOLDER) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            raise ValueError(
                f"No images found in {config.DATABASE_FOLDER}"
            )
            
        # Load or generate features
        if not os.path.exists(config.FEATURES_FILE):
            self.logger.info("Generating features...")
            from extract_features import extract_features_from_directory
            features_df = extract_features_from_directory(config.DATABASE_FOLDER)
            features_df.to_csv(config.FEATURES_FILE, index=False)
            
        self.database.load_features()
        
    async def process_uploaded_image_async(self, img_path: str) -> List[Tuple[str, float]]:
        """Async version of process_uploaded_image"""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                self.executor,
                self.process_uploaded_image,
                img_path
            )
        except Exception as e:
            self.processor.logger.error(f"Async image processing error: {str(e)}")
            return []
            
    def process_uploaded_image(self, img_path: str) -> List[Tuple[str, float]]:
        """Process uploaded image and return recommendations"""
        try:
            # Preprocess image
            img = self.processor.preprocess_image(img_path)
            if img is None:
                return []
                
            # Extract features
            features = self.processor.extract_features(img)
            
            # Find similar images
            return self.database.find_similar_images(features)
        except Exception as e:
            self.processor.logger.error(f"Error processing image: {str(e)}")
            return []

# Initialize Flask app and config
app = Flask(__name__)
config = AppConfig()
config.validate()

@app.route('/', methods=['GET', 'POST'])
async def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Process image and get recommendations
            recommender = ImageRecommender(config)
            recommendations = await recommender.process_uploaded_image_async(filepath)
            
            return render_template('result.html', 
                                 recommendations=recommendations,
                                 original_image=filename)
    
    return render_template('upload.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
