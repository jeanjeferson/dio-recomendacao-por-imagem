import os
import shutil
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import cv2
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

class ImageProcessor:
    """Handles image processing and feature extraction"""
    
    def __init__(self):
        self.img_size = (224, 224)
        self.model = self._load_vgg16_model()
        self.logger = self._setup_logger()
        
    def _load_vgg16_model(self):
        """Load and return VGG16 model without classification layer"""
        base_model = VGG16(weights='imagenet', 
                          include_top=False, 
                          input_shape=(224, 224, 3))
        return Model(inputs=base_model.input, 
                    outputs=base_model.output)
        
    def _setup_logger(self):
        """Configure and return logger"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
        
    def preprocess_image(self, img_path: str) -> np.ndarray:
        """Preprocess image for VGG16 model"""
        img = cv2.imread(img_path)
        if img is None:
            self.logger.error(f"Failed to read image: {img_path}")
            return None
            
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return preprocess_input(img)
        
    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """Extract features from preprocessed image"""
        features = self.model.predict(img)
        return features.reshape(features.shape[0], -1)

class ImageDatabase:
    """Manages the image database and similarity search"""
    
    def __init__(self):
        self.features = None
        self.labels = None
        self.logger = logging.getLogger(__name__)
        
    def load_features(self, csv_path: str = 'image_features.csv'):
        """Load features from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            self.features = df.iloc[:, :-1].values
            self.labels = df['label'].values
            self.logger.info(f"Loaded {len(self.labels)} image features")
        except Exception as e:
            self.logger.error(f"Failed to load features: {str(e)}")
            raise
            
    def find_similar_images(self, query_features: np.ndarray, top_n: int = 5) -> List[Tuple[str, float]]:
        """Find similar images using cosine similarity"""
        if self.features is None:
            self.logger.error("Features not loaded")
            return []
            
        similarities = cosine_similarity(query_features, self.features)
        indices = np.argsort(similarities[0])[::-1][1:top_n+1]
        
        return [(self.labels[i], float(similarities[0][i])) 
                for i in indices]


class ImageRecommender:
    """Main class for image recommendation system"""
    
    def __init__(self):
        self.processor = ImageProcessor()
        self.database = ImageDatabase()
        
        # Verificar se há imagens no diretório database
        database_dir = app.config['DATABASE_FOLDER']
        if not os.path.exists(database_dir):
            os.makedirs(database_dir)
            raise FileNotFoundError(
                f"O diretório {database_dir} está vazio. "
                "Por favor, adicione algumas imagens antes de continuar."
            )
            
        image_files = [f for f in os.listdir(database_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            raise ValueError(
                f"Nenhuma imagem encontrada em {database_dir}. "
                "Por favor, adicione algumas imagens antes de continuar."
            )
        
        # Verificar se o arquivo de características existe
        if not os.path.exists('image_features.csv'):
            print("Arquivo de características não encontrado. Gerando características...")
            from extract_features import extract_features_from_directory
            features_df = extract_features_from_directory(database_dir)
            features_df.to_csv('image_features.csv', index=False)
            print("Características geradas com sucesso!")
        
        self.database.load_features()
        
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATABASE_FOLDER'] = 'static/database'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # limite de 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATABASE_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Processar imagem e obter recomendações
            recommender = ImageRecommender()
            recommendations = recommender.process_uploaded_image(filepath)
            
            return render_template('result.html', 
                                 recommendations=recommendations,
                                 original_image=filename)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
