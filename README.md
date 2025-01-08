# Sistema de Recomendação por Imagens

Sistema que utiliza deep learning para recomendar imagens similares baseado em uma imagem de consulta.

## Configuração com Docker

### Pré-requisitos
- Docker
- Docker Compose

### Executando o Sistema

1. Clone o repositório:
```bash
git clone <repository-url>
cd dio-recomendacao-por-imagem
```

2. Construa e inicie os containers:
```bash
docker-compose up --build
```

3. Acesse o sistema em: http://localhost

## Configuração Local (Sem Docker)

### Pré-requisitos
- Python 3.12
- pip

### Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute o servidor:
```bash
python run.py
```

3. Acesse o sistema em: http://localhost:5000

## Uso

1. **Upload da Base de Dados**
   - Clique em "Processar Base de Dados"
   - Selecione o arquivo ZIP contendo as imagens originais
   - Aguarde o processamento

2. **Buscar Imagens Similares**
   - Clique em "Buscar Similares"
   - Selecione o arquivo ZIP contendo as imagens de teste
   - O sistema mostrará as imagens mais similares encontradas

## Estrutura de Diretórios

- `/app` - Código fonte da aplicação Flask
- `/models` - Modelos e índices gerados
- `/uploads` - Arquivos temporários de upload
- `/temp` - Arquivos temporários de processamento
- `/static` - Arquivos estáticos e imagens processadas

## Notas

- O sistema suporta uploads de arquivos grandes (até 500MB)
- Formatos suportados: JPG, JPEG, PNG
- As imagens são processadas usando o modelo ResNet50
- A similaridade é calculada usando o algoritmo Annoy

## Solução de Problemas

Se encontrar o erro "Request Entity Too Large":
1. Use a versão Docker que já está configurada para lidar com arquivos grandes
2. Ou ajuste as configurações do seu servidor web para aumentar o limite de upload
