from transformers import AutoTokenizer # passa o texto pros tokens que o BERT espera
from transformers import AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import time
import nltk # pra dividir em sentenças
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize
import torch

#==================================================================
#Configs
import logging
import warnings
from transformers.utils import logging as hf_logging
import os

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"

#==================================================================
#Funções gerais

def chunking(text):
    if CHUNK_MODE == 'sentences':
        parts = []
        for sentence in sent_tokenize(text):
            sentence = sentence.strip()
            if sentence:
                parts.append(sentence)
    elif CHUNK_MODE == 'lines':
        parts = []
        for lines in text.splitlines():
            lines = lines.strip()
            if lines:
                parts.append(lines)
    
    total_logits = torch.zeros(model.config.num_labels)
    total_weights = 0.0    

    for part in parts:
        encoding = tokenizer(
            part,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
    with torch.no_grad():
        output = model(**encoding).logits.squeeze(0)
    weight = float(encoding['attention_mask'].sum().item()) # peso baseado no tamanho do chunk
    total_logits += output * weight
    total_weights += weight

    if total_weights == 0:
        total_weights = 1.0  # p evitar divisao p 0
    averaged_logits = total_logits / total_weights

    probabilities = torch.sigmoid(averaged_logits).numpy()

    one_chunk = []
    for i in range(len(LABELS)):
        one_chunk.append({
            'label': LABELS[i],
            'score': float(probabilities[i])
        })

    return [one_chunk]

def get_all_emotions(results):
    # pega tds as emocoes
    labels = []
    for i in results[0]:
        labels.append(i['label'])
    return labels

def get_average_emotions_scores(labels, results):
    # soma a media dos scores dos chunks
    soma = {}
    for label in labels:
        soma[label] = 0.0

    for result in results:
        for i in result:
            soma[i['label']] += i['score']

    # pega todas as medias das emocoes
    results_qtt = len(results)
    average = {}
    for label in labels:
        average[label] = soma[label] / results_qtt

    return average

# pega a emoção mais demarcada através do score dela
def get_top1_emotion(labels, average):
    top1_label = None
    top1_score = 0.0
    for label in labels:
        if average[label] > top1_score:
            top1_score = average[label]
            top1_label = label
    return top1_label, top1_score

#==================================================================
start_time = time.time()

FOLDER = 'Letras/'
CHUNK_MODE = 'sentences' #lines ou sentences (ver qual fica melhor)
MAX_LENGTH = 256 
STRIDE = 64 # isso aqui é o overlap entre chunks
TOPK = 3 # quantas emoções pegar por chunk
MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions"
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    problem_type="multi_label_classification",
)

# já faz automaticamente a tokenizacao pro modelo específico
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# salva numa ordem estavel as emocoes
num_labels = model.config.num_labels
LABELS = [model.config.id2label[i] for i in range(num_labels)]
rows = []
labels_ref = None

#-------------------------------------------------------------------
# cria o 1o dataframe com todas as emocoes

# percorre os arquivos de texto na pasta
for file_name in os.listdir(FOLDER):
    path = os.path.join(FOLDER, file_name)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    print("Analisando:", os.path.splitext(file_name)[0])

    if len(text) < 10:
        os.remove(path)
        print(f"Arquivo {file_name} removido por ter menos de 10 caracteres.")
        continue

    results = chunking(text)
    labels = get_all_emotions(results)
    if labels_ref is None:
        labels_ref = labels
    average = get_average_emotions_scores(labels, results)
    top1_label, top1_score =get_top1_emotion(labels, average)
    
    # cria a linha do dataframe
    row = {'musica': os.path.splitext(file_name)[0],
           'top1_emocao': f'{top1_label}({top1_score:.3f})',
           **average
           }
    # adiciona as colunas de emocoes
    for label in labels_ref:
        row[label] = average[label]
    rows.append(row)

# cria o dataframe final
columns = ['musica', 'top1_emocao']
for label in labels_ref:
    columns.append(label)
df = pd.DataFrame(rows, columns=columns)
df.to_csv(f"all_emotions_PTBR.csv", index=False)

#-------------------------------------------------------------------
# filtra as emocoes para pegar apenas as top 10 mais frequentes


# fiz isso pois são muitas emocoes e o score fica muito dividido
df = pd.read_csv("all_emotions_PTBR.csv")
emotions_columns = df.columns.tolist()[2:]
means = df[emotions_columns].mean().sort_values(ascending=False)
top10_emotions = means.head(10).index.tolist()

df_top10 = df[['musica', 'top1_emocao'] + top10_emotions].copy()
row_sums = df_top10[top10_emotions].sum(axis=1).replace(0, 1.0)
df_top10[top10_emotions] = df_top10[top10_emotions].div(row_sums, axis=0)
df_top10.to_csv("top10_emotions_PTBR.csv", index=False)

end_time = time.time()
print(f"Tempo de execucao: {end_time - start_time:.2f} segundos")
