import os
import spacy

def calculate_metrics(text, expected_entities):
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

    # Calculate True Positives (correctly predicted entities)
    true_positives = set(predicted_entities).intersection(set(expected_entities))

    # False Positives (predicted but not in expected)
    false_positives = set(predicted_entities) - set(expected_entities)

    # False Negatives (expected but not predicted)
    false_negatives = set(expected_entities) - set(predicted_entities)

    # Calculate Precision
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0.0

    # Calculate Recall
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0.0

    # Calculate F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Convert to percentages for better readability
    precision_percentage = precision * 100
    recall_percentage = recall * 100
    f1_score_percentage = f1_score * 100

    return {
        "precision": precision_percentage,
        "recall": recall_percentage,
        "f1_score": f1_score_percentage,
        "true_positives": list(true_positives),
        "false_positives": list(false_positives),
        "false_negatives": list(false_negatives)
    }

# Example usage
example = {
    "text": "Em uma manhã tranquila na floresta, o leão, com sua juba dourada, caminhava lentamente enquanto observava o elefante tomando água do rio. O macaco saltava de galho em galho, fazendo brincadeiras com a onça-pintada, que o observava com seus olhos atentos. No chão, a tartaruga se movia com paciência, e perto de um arbusto, a lebre corria apressada, seguida pela raposa astuta. A águia sobrevoava a cena, com suas asas largas cortando o vento, enquanto o crocodilo se aquecia sob o sol. O coelho estava escondido em uma toca, enquanto a coruja vigiava a floresta à noite. A cobra deslizou pelo terreno, e a girafa, com seu pescoço longo, olhava por cima das árvores, ao lado de um grupo de zebras pastando tranquilamente. Não muito longe dali, o lobo uivava à distância, e o urso caminhava perto do riacho. No alto, o beija-flor pairava sobre uma flor, enquanto o pinguim nadava nas águas geladas de um lago remoto.",
    "expected_entities": [
        "leão", "elefante", "macaco", "onça-pintada", "tartaruga", "lebre",
        "raposa", "águia", "crocodilo", "coelho", "coruja", "cobra",
        "girafa", "zebra", "lobo", "urso", "beija-flor", "pinguim"
    ]
}

metrics = calculate_metrics(example["text"], example["expected_entities"])
print("Model Metrics:")
print(f"Precision: {metrics['precision']:.2f}%")
print(f"Recall: {metrics['recall']:.2f}%")
print(f"F1-Score: {metrics['f1_score']:.2f}%")
print("True Positives:", metrics["true_positives"])
print("False Positives:", metrics["false_positives"])
print("False Negatives:", metrics["false_negatives"])