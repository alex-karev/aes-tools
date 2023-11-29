import json
import nltk.data
import pandas as pd
import statistics
import os

# Class for manipulating datasets for AES more easily
class AESData:
    # Constructor. Uses a special json card describing dataset
    def __init__(self, dataset_path, printer_format="github"):
        self.printer_format = printer_format
        # Load dataset card
        self.p = {}
        with open(dataset_path, "r") as f:
            self.p = json.loads(f.read())
        f.close()
        # Load data
        self.d = []
        path = os.path.join(os.path.dirname(dataset_path), self.p["path"])
        with open(path, "r", errors='replace') as f:
            for line in f.readlines():
                self.d.append(line.split("\t"))
        f.close()
        # Remove the first line
        if self.p["skip_first_line"]:
            self.d.pop(0)
        # Shortcut for column data
        self.ESSAY_COL = self.p["columns"]["essay"]
        self.SCORE_COL = self.p["columns"]["score"]
        self.PROMPT_COL = self.p["columns"]["prompt"]
        # Load tokenizer
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    # Meta
    def get_dataset_path(self): return self.p["path"]
    def get_dataset_author(self): return self.p["author"]
    def get_dataset_publisher(self): return self.p["publisher"]
    def get_dataset_title(self): return self.p["title"]
    def get_dataset_year(self): return self.p["year"]
    def get_dataset_url(self): return self.p["url"]
    def get_prompt_min_score(self, prompt): return self.p["prompts"][prompt-1]["min_score"]
    def get_prompt_max_score(self, prompt): return self.p["prompts"][prompt-1]["max_score"]
    def get_prompt_genre(self, prompt): return self.p["prompts"][prompt-1]["genre"]
    
    # Getters
    def get_essay(self, id):
        return self.d[id][self.ESSAY_COL]
    def get_essay_as_arr(self, id):
        essay = []
        for sentence in self.tokenizer.tokenize(self.get_essay(id)):
            essay.append(sentence.split())
        return essay
        sentences = self.get_essay(id).strip().replace("\n","").split(".")
        for sentence in sentences:
            essay.append(sentence.split())
        return essay
    def get_score(self, id):
        return int(self.d[id][self.SCORE_COL].replace("\n",""))
    def get_prompt(self, id):
        return int(self.d[id][self.PROMPT_COL].replace("\n",""))
    def get_score_norm(self, id):
        prompt = self.get_prompt(id)
        min_score = float(self.get_prompt_min_score(prompt))
        max_score = float(self.get_prompt_max_score(prompt))
        score = float(self.get_score(id))
        return (score - min_score) / (max_score - min_score)
    def get_score_percent(self, id):
        return round(self.get_score_norm(id)*100.0)
    def get_prompt_essays(self, prompt):
        ids = []
        for i in range(len(self.d)):
            if self.get_prompt(i) == prompt:
                ids.append(i)
        return ids
    
    # Counters
    def count_prompts(self):
        return len(self.p["prompts"])
    def count_essays(self):
        return len(self.d)
    def count_pompt_essays(self, prompt):
        return len(self.get_prompt_essays(prompt))

    # Statistics
    def get_avg_words(self, prompt):
        avg_words = 0
        n = 0
        for id in self.get_prompt_essays(prompt):
            #avg_words += len(nltk.word_tokenize(self.get_essay(id)))
            avg_words += len(self.get_essay(id).split(" "))
            n += 1
        return float(avg_words)/float(n)
    def get_stdev_words(self, prompt):
        words = []
        for id in self.get_prompt_essays(prompt):
            words.append(len(self.get_essay(id).split(" ")))
        return statistics.stdev(words)
    def get_avg_score(self, prompt):
        avg_score = 0
        n = 0
        for id in self.get_prompt_essays(prompt):
            avg_score += self.get_score(id)
            n += 1
        return float(avg_score)/float(n)
    def get_stdev_score(self, prompt):
        scores = []
        for id in self.get_prompt_essays(prompt):
            scores.append(self.get_score_norm(id))
        return statistics.stdev(scores)
    def get_avg_score_norm(self, prompt):
        min_score = float(self.get_prompt_min_score(prompt))
        max_score = float(self.get_prompt_max_score(prompt))
        score = self.get_avg_score(prompt)
        return (score-min_score)/(max_score-min_score)
    def get_min_score(self, prompt):
        min_score = 1000
        for id in self.get_prompt_essays(prompt):
            score = self.get_score(id)
            if score < min_score:
                min_score = score 
        return min_score
    def get_max_score(self, prompt):
        max_score = -1000
        for id in self.get_prompt_essays(prompt):
            score = self.get_score(id)
            if score > max_score:
                max_score = score 
        return max_score

    # Info
    def get_meta(self):
        return {
            "Title": self.get_dataset_title(),
            "URL": self.get_dataset_url(),
            "Author": ", ".join(self.get_dataset_author()),
            "Publisher": self.get_dataset_publisher(),
            "Year": self.get_dataset_year(),
            "Prompts Number": self.count_prompts(),
            "Essays Number":  self.count_essays()
        }
    def get_prompt_meta(self, prompt):
        return {
            "Prompt": prompt,
            "Genre": self.get_prompt_genre(prompt),
            "Essays": self.count_pompt_essays(prompt),
            "Avg. Words": self.get_avg_words(prompt),
            "Stdev. Words": self.get_stdev_words(prompt),
            "Min Score": self.get_prompt_min_score(prompt),
            "Max Score": self.get_prompt_max_score(prompt),
            "Avg. Score": self.get_avg_score(prompt),
            "Avg. Score (Norm.)": self.get_avg_score_norm(prompt),
            "Stdev. Score (Norm.)": self.get_stdev_score(prompt)
        }
    def get_stats(self):
        output = []
        for i in range(1,self.count_prompts()+1):
            output.append(self.get_prompt_meta(i))
        return output

    # Printers
    def print_meta(self):
        meta = self.get_meta()
        keys = list(meta.keys())
        values = list(meta.values())
        df = pd.DataFrame({keys[0]: keys[1:], values[0]: values[1:]})
        print(df.to_markdown(tablefmt=self.printer_format,index=False),"\n")
    def print_prompts(self):
        df = pd.json_normalize(self.get_stats())
        print(df.to_markdown(floatfmt=".2f",tablefmt=self.printer_format,index=False,),"\n")
    def print_info(self):
        self.print_meta()
        self.print_prompts()
        
    