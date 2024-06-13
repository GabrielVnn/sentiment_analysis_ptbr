from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Carregar o modelo e o tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)

classes = ["Muito negativo", "Negativo", "Neutro", "Positivo", "Muito positivo"]


def sentiment_analysis(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    sentiment = torch.argmax(probabilities, dim=-1).item()
    return classes[sentiment], probabilities


# Texto para análise de sentimento
text = "Estou animado com meus estudos de AI!"
sentiment, probabilities = sentiment_analysis(text)

# Imprimir o resultado
print(f"Sentimento: {sentiment}")
print(f"Probabilidades: {probabilities}")
