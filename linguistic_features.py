#!.env/bin/python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

ORIGINAL_DATASET_PATH = "datasets/ASAP.json"
DATASET_PATH = "linguistic_features.csv"
PROMPT_NUM = 8
TRAIN_PROMPTS = [1,3,7]
TEST_PROMPTS = [2,4,5,6,8]

# Generate dataset for training
def generate(dataset_path: str, save_path: str):
    from AESData import AESData
    from AESLinguisticFeatures import AESLinguisticFeatures
    data = AESData(dataset_path)
    features = AESLinguisticFeatures(data)
    features.generate_dataset(save_path)
#generate(ORIGINAL_DATASET_PATH, DATASET_PATH)

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Train model
train_data = df[~df["prompt"].isin(TRAIN_PROMPTS)].drop(["prompt","id"], axis=1)
x = train_data.drop("score", axis=1)
y = train_data["score"]
reg = LinearRegression()
reg.fit(x, y)

# Loop through each test prompt
for prompt in TEST_PROMPTS:
    # Prepare data
    test_data = df[~df["prompt"].isin([prompt])].drop(["prompt","id"], axis=1)
    x = test_data.drop("score", axis=1)
    y = test_data["score"]

    # Predict score
    y_predict = reg.predict(x)

    # Convert scores
    original_percent = [round(score*100) for score in y]
    predict_percent = [round(score*100) for score in y_predict]
    original_ten = [round(score*10) for score in y]
    predict_ten = [round(score*10) for score in y_predict]

    # Calculate QWK
    qwk_perecent = cohen_kappa_score(original_percent, predict_percent, labels=None, weights="quadratic", sample_weight=None)
    qwk_ten = cohen_kappa_score(original_ten, predict_ten, labels=None, weights="quadratic", sample_weight=None)

    # Print output
    print("Prompt {}:\n\tQWK (n/100): {:.4f}\n\tQWK (n/10): {:.4f}".format(prompt, qwk_perecent, qwk_ten))

    #cm=confusion_matrix(original_ten,predict_ten)
    #plt.figure(figsize = (10,7))
    #sn.heatmap(cm, annot=True)
    #plt.show()