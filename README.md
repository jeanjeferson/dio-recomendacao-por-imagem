# 🖼️ Sistema de Recomendação de Imagens

Bem-vindo ao nosso sistema de recomendação de imagens! 👋 Este é um sistema inteligente que usa aprendizado profundo para encontrar imagens semelhantes à que você enviar. Vamos configurar? 🚀

## 📋 Pré-requisitos

Antes de começar, você vai precisar de:

- Python 3.7 ou superior 🐍
- Gerenciador de pacotes pip 📦

## 🛠️ Instalação

Vamos configurar o ambiente passo a passo:

1. **Clone o repositório**  
   Primeiro, faça o download do projeto:
   ```bash
   git clone https://github.com/seu-usuario/recomendacao-imagem.git
   cd recomendacao-imagem
   ```

2. **Crie um ambiente virtual**  
   Vamos isolar as dependências do projeto:
   ```bash
   python -m venv venv
   # No Windows:
   venv\Scripts\activate
   # No Mac/Linux:
   source venv/bin/activate
   ```

3. **Instale as dependências**  
   Agora, instale tudo que precisamos:
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Estrutura do Projeto

Aqui está o que cada pasta contém:

```
📁 static/
   ├── 📁 database/    # Aqui ficam as imagens para recomendação
   └── 📁 uploads/     # Onde as imagens enviadas são armazenadas
📁 templates/          # Templates HTML da interface web
📄 image_features.csv  # Características pré-calculadas das imagens
📄 recomendacao.py     # O coração do sistema
```

## 🚀 Executando o Sistema

Vamos colocar o sistema para funcionar!

1. **Prepare as imagens**  
   Coloque suas imagens na pasta `static/database/`

2. **Inicie o servidor**  
   No terminal, execute:
   ```bash
   python recomendacao.py
   ```

3. **Acesse o sistema**  
   Abra seu navegador e visite:  
   http://localhost:5000

## 🖥️ Como Usar

É super simples! 😊

1. Acesse http://localhost:5000
2. Escolha uma imagem usando o seletor de arquivos
3. O sistema mostrará as imagens mais parecidas da base de dados

## 📦 Dependências

Estes são os pacotes Python que usamos:

- Flask 🌐
- TensorFlow 🧠
- NumPy 🔢
- Pandas 🐼
- Scikit-learn 📊
- OpenCV 📷
- Werkzeug 🛠️

## ⚠️ Importante!

- As características das imagens estão pré-calculadas no arquivo `image_features.csv`
- Formatos suportados: JPG, JPEG, PNG
- Tamanho máximo de upload: 16MB

Pronto para começar? Vamos lá! 🎉
