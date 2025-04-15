import os
import spacy

# Obtém o diretório atual do script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define o caminho relativo para a pasta do modelo
model_path = os.path.join(base_dir, 'models', 'model_animals')

# Carrega o modelo do caminho relativo
nlp = spacy.load(model_path)

# Testando o modelo com uma entrada
doc = nlp("")

# Imprime as entidades detectadas
for entidade in doc.ents:
    print("Entity:", entidade.text, "| Label:", entidade.label_)
