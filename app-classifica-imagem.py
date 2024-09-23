from urllib.request import urlopen
from PIL import Image
import timm
import torch
import requests
import json

# ---------------------------
# 1. Carregamento da Imagem
# ---------------------------

# URL da imagem a ser classificada
image_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'

# Abrir a imagem a partir da URL usando PIL
try:
    img = Image.open(urlopen(image_url)).convert('RGB')  # Converter para RGB para garantir compatibilidade
    print("Imagem carregada com sucesso.")
except Exception as e:
    print(f"Erro ao carregar a imagem: {e}")
    exit()

# ---------------------------
# 2. Carregamento do Modelo
# ---------------------------

# Nome do modelo pré-treinado a ser usado
model_name = 'resnet50.a1_in1k'

# Carregar o modelo pré-treinado usando timm
try:
    model = timm.create_model(model_name, pretrained=True)
    model = model.eval()  # Colocar o modelo em modo de avaliação
    print(f"Modelo '{model_name}' carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    exit()

# -----------------------------------
# 3. Preparação das Transformações
# -----------------------------------

# Obter as configurações específicas do modelo (como normalização e redimensionamento)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Aplicar as transformações à imagem e adicionar uma dimensão de batch
input_tensor = transforms(img).unsqueeze(0)  # Forma: [1, C, H, W]

# -----------------------------------
# 4. Inferência com o Modelo
# -----------------------------------

# Realizar a inferência sem calcular gradientes para maior eficiência
with torch.no_grad():
    output = model(input_tensor)  # Saída: [1, 1000] logits

# -----------------------------------
# 5. Cálculo das Probabilidades
# -----------------------------------

# Aplicar softmax para converter logits em probabilidades
probabilities = output.softmax(dim=1) * 100  # Convertendo para porcentagem

# -----------------------------------
# 6. Seleção das Top 5 Previsões
# -----------------------------------

# Obter as top 5 probabilidades e seus índices de classe correspondentes
top5_probabilities, top5_class_indices = torch.topk(probabilities, k=5)

# -----------------------------------
# 7. Mapeamento de Índices para Nomes de Classes
# -----------------------------------

# URL do arquivo JSON que mapeia índices de classe para nomes de classe
class_index_url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

# Baixar o arquivo JSON com os mapeamentos das classes
try:
    response = requests.get(class_index_url)
    imagenet_class_index = response.json()
    print("Mapeamento de classes carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o mapeamento das classes: {e}")
    exit()

# Função para obter o nome da classe a partir do índice
def get_class_name(idx):
    return imagenet_class_index[str(idx)][1]  # Retorna o nome da classe

# Obter os nomes das top 5 classes
top5_class_names = [get_class_name(idx.item()) for idx in top5_class_indices[0]]

# -----------------------------------
# 8. Impressão dos Resultados
# -----------------------------------

print("\n=== Top 5 Previsões ===")
for i in range(5):
    class_name = top5_class_names[i]
    probability = top5_probabilities[0][i].item()
    print(f"{i+1}. {class_name}: {probability:.2f}%")
