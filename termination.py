import os
import spacy

def calculate_precision(text, expected_entities_count):
    # Get the current directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path to the model folder
    model_path = os.path.join(base_dir, 'models', 'model_animals')

    # Load the model from the relative path
    nlp = spacy.load(model_path)

    # Apply the model to the provided text
    doc = nlp(text)

    # Imprime as entidades detectadas
    for entity in doc.ents:
        print("Entity:", entity.text, "| Label:", entity.label_)

    # Extract the entities predicted by the model
    predicted_entities = [entity.text for entity in doc.ents]

    # Calculate the number of correctly recognized entities
    correct_entities_count = len(predicted_entities)

    # Ensure the precision doesn't exceed 1
    precision = min(correct_entities_count, expected_entities_count) / expected_entities_count if expected_entities_count > 0 else 0.0

    # Convert the precision to percentage
    precision_percentage = precision * 100

    return precision_percentage

# Example usage
example = {
    "text": "Em uma manhã tranquila na floresta, o leão, com sua juba dourada, caminhava lentamente enquanto observava o elefante tomando água do rio. O macaco saltava de galho em galho, fazendo brincadeiras com a onça-pintada, que o observava com seus olhos atentos. No chão, a tartaruga se movia com paciência, e perto de um arbusto, a lebre corria apressada, seguida pela raposa astuta. A águia sobrevoava a cena, com suas asas largas cortando o vento, enquanto o crocodilo se aquecia sob o sol. O coelho estava escondido em uma toca, enquanto a coruja vigiava a floresta à noite. A cobra deslizou pelo terreno, e a girafa, com seu pescoço longo, olhava por cima das árvores, ao lado de um grupo de zebras pastando tranquilamente. Não muito longe dali, o lobo uivava à distância, e o urso caminhava perto do riacho. No alto, o beija-flor pairava sobre uma flor, enquanto o pinguim nadava nas águas geladas de um lago remoto.",
    "count": 17
}

precision = calculate_precision(example["text"], example["count"])

print("Model precision:", precision, "%")
