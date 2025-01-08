import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import cv2
import logging

def extract_features_from_directory(directory):
    """Extrai características de todas as imagens em um diretório"""
    logger = logging.getLogger(__name__)
    
    # Verificar se o diretório existe
    if not os.path.exists(directory):
        logger.error(f"Diretório {directory} não encontrado!")
        raise FileNotFoundError(f"Diretório {directory} não encontrado!")
    
    # Verificar se há imagens no diretório
    image_files = [f for f in os.listdir(directory) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        logger.error(f"Nenhuma imagem encontrada no diretório {directory}")
        raise ValueError(f"Nenhuma imagem encontrada no diretório {directory}")
    
    # Carregar modelo VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    
    features_list = []
    filenames = []
    
    # Processar cada imagem no diretório
    for filename in image_files:
        try:
            filepath = os.path.join(directory, filename)
            
            # Carregar e preprocessar imagem
            img = cv2.imread(filepath)
            if img is None:
                logger.warning(f"Não foi possível ler a imagem: {filepath}")
                continue
                
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            
            # Extrair características
            features = model.predict(img, verbose=0)
            features = features.reshape(1, -1)
            
            features_list.append(features[0])
            filenames.append(filename)
            
        except Exception as e:
            logger.error(f"Erro ao processar {filename}: {str(e)}")
            continue
    
    if not features_list:
        logger.error("Nenhuma característica extraída com sucesso")
        raise ValueError("Nenhuma característica extraída com sucesso")
    
    # Criar DataFrame com características
    features_array = np.array(features_list)
    feature_columns = [f'feature_{i}' for i in range(features_array.shape[1])]
    df = pd.DataFrame(features_array, columns=feature_columns)
    df['label'] = filenames
    
    return df

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Extrair características das imagens no diretório database
        database_dir = 'static/database'
        logger.info("Iniciando extração de características...")
        features_df = extract_features_from_directory(database_dir)
        
        # Salvar características em CSV
        features_df.to_csv('image_features.csv', index=False)
        logger.info(f"Características de {len(features_df)} imagens salvas em image_features.csv")
        
    except Exception as e:
        logger.error(f"Erro durante a extração de características: {str(e)}")
        exit(1) 