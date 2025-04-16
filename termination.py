import os
import spacy

def calculate_precision(text, expected_entities):
    # Get the current directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the model folder
    model_path = os.path.join(base_dir, 'models', 'model_animals')

    # Load the model from the relative path
    nlp = spacy.load(model_path)

    # Apply the model to the provided text
    doc = nlp(text)

    # Print the detected entities
    print("Entidades detectadas:")
    for entity in doc.ents:
        print("Entity:", entity.text, "| Label:", entity.label_)

    # Normalize entities for comparison (e.g., lowercase, strip whitespace)
    predicted_entities = [ent.text.strip().lower() for ent in doc.ents]
    expected_entities = [ent.strip().lower() for ent in expected_entities]

    # Calculate correct predictions (intersection)
    correct_entities = [ent for ent in predicted_entities if ent in expected_entities]

    # Avoid counting duplicates twice
    correct_entities_set = set(correct_entities)

    # Calculate precision
    precision = len(correct_entities_set) / len(expected_entities) if expected_entities else 0.0

    # Convert to percentage
    precision_percentage = precision * 100

    return precision_percentage

# Exemplo de uso
example = {
    "text": "Em uma manhã tranquila na floresta, o leão, com sua juba dourada, caminhava lentamente enquanto observava o elefante tomando água do rio. O macaco saltava de galho em galho, fazendo brincadeiras com a onça-pintada, que o observava com seus olhos atentos. No chão, a tartaruga se movia com paciência, e perto de um arbusto, a lebre corria apressada, seguida pela raposa astuta. A águia sobrevoava a cena, com suas asas largas cortando o vento, enquanto o crocodilo se aquecia sob o sol. O coelho estava escondido em uma toca, enquanto a coruja vigiava a floresta à noite. A cobra deslizou pelo terreno, e a girafa, com seu pescoço longo, olhava por cima das árvores, ao lado de um grupo de zebras pastando tranquilamente. Não muito longe dali, o lobo uivava à distância, e o urso caminhava perto do riacho. No alto, o beija-flor pairava sobre uma flor, enquanto o pinguim nadava nas águas geladas de um lago remoto.",
    "expected_entities": [
        "leão", "elefante", "macaco", "onça-pintada", "tartaruga", "lebre",
        "raposa", "águia", "crocodilo", "coelho", "coruja", "cobra",
        "girafa", "zebra", "lobo", "urso", "beija-flor", "pinguim"
    ]
}

precision = calculate_precision(example["text"], example["expected_entities"])
print("Model precision:", precision, "%")
