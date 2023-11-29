#!.env/bin/python
from AESData import AESData
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import cohen_kappa_score

# It is completely useless

# Global variables
DATASET_PATH = "datasets/ASAP.json"

# Load dataset
d = AESData(DATASET_PATH)
d.print_info()

# Find best essays in prompt
prompt = 1
references = []
for id in d.get_prompt_essays(prompt):
    if d.get_score(id) == d.get_prompt_max_score(prompt):
        references = d.get_essay_as_arr(id)
        break

# Choose random essay as a template
# Score essays using BLEU
original_scores = []
predicted_scores = []
for id in d.get_prompt_essays(prompt):
    candidates = d.get_essay_as_arr(id)
    if candidates == references:
        continue
    predicted_score = 0
    for s in candidates:
        predicted_score += sentence_bleu(references, s, weights=(0.9,0.1,0,0))
    predicted_score /= len(candidates)
    predicted_score = round(predicted_score*100.0)+30
    original_score = d.get_score_percent(id)
    print("{:.2f} : {:.2f}".format(original_score, predicted_score))
    original_scores.append(original_score)
    predicted_scores.append(predicted_score)

qwk = cohen_kappa_score(original_scores, predicted_scores, labels=None, weights="quadratic", sample_weight=None)
print("QWK: {qwk}".format(qwk = qwk))
