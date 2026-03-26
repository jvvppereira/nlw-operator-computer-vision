# Use uma imagem oficial do Python slim (leve e baseada em Debian)
FROM python:3.12-slim

# Instala as dependências do sistema necessárias para OpenCV e MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia o arquivo de dependências primeiro (otimiza o cache do Docker)
COPY requirements.txt .

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o conteúdo do projeto para o container
COPY . .

# Expõe a porta que o Render costuma usar (embora ele gerencie isso)
EXPOSE 10000

# Comando para rodar o app usando Uvicorn, respeitando a porta do Render
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
