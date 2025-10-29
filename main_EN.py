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

# cria o dataframe e salva em .csv
def create_dataframe(top1_label, top1_score, average):
    df = pd.DataFrame([{
        'musica': 'dualmente',
        'top1_emocao': f'{top1_label}({top1_score:.3f})',
        **average
    }])
    df.to_csv(f"emocoes.csv", index=False)

#==================================================================
start_time = time.time()

FOLDER = 'Lyrics/'
CHUNK_MODE = 'sentences' #lines ou sentences (ver qual fica melhor)
MAX_LENGTH = 256 
STRIDE = 64 # isso aqui é o overlap entre chunks
TOPK = 3 # quantas emoções pegar por chunk
MODEL_NAME = "SamLowe/roberta-base-go_emotions"
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
df.to_csv(f"all_emotions_EN.csv", index=False)

#-------------------------------------------------------------------
# filtra as emocoes para pegar apenas as top 10 mais frequentes

df = pd.read_csv("all_emotions_EN.csv")

# tira espaços e padroniza nomes de colunas
renamed = {}
for c in df.columns:
    renamed[c] = c.strip()
df = df.rename(columns=renamed)

# remove colunas duplicadas
dup_mask = df.columns.duplicated(keep="first")
df = df.loc[:, ~dup_mask]

# lista de colunas de emocoes
all_cols = df.columns.tolist()
emotions_columns = []
i = 2
while i < len(all_cols):
    emotions_columns.append(all_cols[i])
    i += 1

for c in emotions_columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# médias por emoção
means = df[emotions_columns].mean()
means = means.sort_values(ascending=False)

# remove neutral
if "neutral" in means.index:
    means = means.drop("neutral", errors="ignore")

# pega as 10 mais altas
top10_emotions_no_neutral = []
for lbl in means.head(10).index:
    top10_emotions_no_neutral.append(lbl)

# montar colunas finais sem duplicar
keep_cols = ["musica", "top1_emocao"]
for lbl in top10_emotions_no_neutral:
    if lbl not in keep_cols:
        keep_cols.append(lbl)

df_top10 = df[keep_cols].copy()

# renormaliza somente pelas não-neutras
row_sums = df_top10[top10_emotions_no_neutral].sum(axis=1)
row_sums = row_sums.replace(0, 1.0)
for i in range(len(df_top10)):
    denominador = row_sums.iat[i]
    for lbl in top10_emotions_no_neutral:
        df_top10.at[i, lbl] = float(df_top10.at[i, lbl]) / float(denominador)

# recalcula top1_emocao ignorando neutral
def top1_no_neutral(row):
    best_label = None
    best_score = -1.0
    for lbl in top10_emotions_no_neutral:
        if lbl in row.index:
            v = row.at[lbl]
            try:
                val = float(v)
            except Exception:
                val = 0.0
        else:
            val = 0.0
        if val > best_score:
            best_score = val
            best_label = lbl
    if best_label is None:
        current = row.get("top1_emocao", "")
        return current
    return f"{best_label}({best_score:.3f})"

df_top10['top1_emocao'] = df_top10.apply(top1_no_neutral, axis=1)

df_top10.to_csv("top10_emotions_EN.csv", index=False)

end_time = time.time()
print(f"Tempo de execucao: {end_time - start_time:.2f} segundos")
