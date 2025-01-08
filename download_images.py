import os
import requests
from PIL import Image
from io import BytesIO
import logging

def download_sample_images():
    """Download algumas imagens de exemplo para o diretório static/database"""
    # URLs de imagens de exemplo do Unsplash (imagens gratuitas e confiáveis)
    image_urls = [
        'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=500',  # Tênis vermelho
        'https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=500',  # Relógio
        'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=500',  # Fones de ouvido
        'https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?w=500',  # Câmera
        'https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=500',  # Óculos
        'https://images.unsplash.com/photo-1585386959984-a4155224a1ad?w=500',  # Bolsa
        'https://images.unsplash.com/photo-1560343090-f0409e92791a?w=500',  # Relógio de pulso
        'https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=500'   # Tênis branco
    ]

    # Criar diretórios necessários
    database_dir = 'static/database'
    uploads_dir = 'static/uploads'
    os.makedirs(database_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    successful_downloads = 0

    # Download e salvar cada imagem
    for i, url in enumerate(image_urls, 1):
        try:
            logger.info(f'Baixando imagem {i} de {len(image_urls)}...')
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Verificar se o conteúdo é realmente uma imagem
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                logger.error(f'URL {url} não retornou uma imagem válida')
                continue

            # Abrir imagem com PIL
            img = Image.open(BytesIO(response.content))
            
            # Converter para RGB se necessário
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Redimensionar imagem para um tamanho razoável
            max_size = (800, 800)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Salvar imagem
            filename = f'product_{i}.jpg'
            filepath = os.path.join(database_dir, filename)
            img.save(filepath, 'JPEG', quality=85)
            
            successful_downloads += 1
            logger.info(f'Download concluído: {filename}')
            
        except requests.RequestException as e:
            logger.error(f'Erro ao baixar imagem {i}: {str(e)}')
        except Exception as e:
            logger.error(f'Erro ao processar imagem {i}: {str(e)}')

    if successful_downloads == 0:
        logger.error('Nenhuma imagem foi baixada com sucesso!')
        return False

    logger.info(f'Download concluído! {successful_downloads} imagens baixadas com sucesso.')
    return True

if __name__ == '__main__':
    print("Iniciando download das imagens de exemplo...")
    if download_sample_images():
        print("Processo concluído com sucesso!")
    else:
        print("Erro: Não foi possível baixar as imagens de exemplo.") 