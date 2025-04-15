import spacy
from spacy.training.example import Example
import os
import json

# Carregar o modelo vazio em português
nlp = spacy.blank('pt')

# Adicionar o componente NER
ner = nlp.add_pipe("ner")

# Definir a classe do rótulo
LABELS = ["ANIMAL"]
for label in LABELS:
    ner.add_label(label)

# Caminho para o arquivo de dados JSON
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jsons', 'animals.json')

# Função para carregar o conteúdo do arquivo JSON
def load_train_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    # Filtra os dados para remover entradas nulas (null)
    annotations = [entry for entry in data['annotations'] if entry is not None]
    
    return annotations

# Carregar os dados de treinamento
train_data = load_train_data(data_path)

# Converte os dados de treinamento em exemplos do spaCy
examples = []
for annotation in train_data:
    text = annotation[0]
    entities = annotation[1].get('entities', [])
    example = Example.from_dict(nlp.make_doc(text), {"entities": entities})
    examples.append(example)

# Inicializa e treina o modelo
nlp.begin_training()
losses = {}
for i in range(100):  # Executar 100 iterações de treinamento
    nlp.update(examples, losses=losses)
    print(losses)  # Imprimir as perdas de treinamento a cada iteração

# Determina o diretório do script atual (onde o arquivo está localizado)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Caminho para salvar o modelo dentro da pasta 'odels'
model_dir = os.path.join(current_dir, 'models', 'model_animals')

# Cria a pasta 'odels' caso ela não exista
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Salva o modelo treinado dentro da pasta 'odels'
nlp.to_disk(model_dir)

print("Modelo salvo com sucesso")