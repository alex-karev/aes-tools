#!.env/bin/python
from AESData import AESData
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import enchant
import re
import os

class AESLinguisticFeatures:
    """Abstraction for linguistic features calculation in essays"""
    
    def __init__(self, data: AESData, difficult_words_path: str = "difficult_words.txt", stopwords_path: str = ""):
        """Constructor. Requires AESData to work with. A path to custom list of difficult words and stop words can be specified (Optional)"""
        self.data = data
        self.sentence_cache = [[] for i in range(self.data.count_essays())]
        self.word_cache = [[] for i in range(self.data.count_essays())]
        self.pos_tags_cache = [[] for i in range(self.data.count_essays())]
        self.total_average_word_length = -1
        self.tokenize_filter = '[^A-Za-z\- ]+'
        self.spellcheck_dict = enchant.Dict("en_US")
        # Load difficult words list
        self.difficult_words_list = []
        if os.path.exists(difficult_words_path):
            with open(difficult_words_path, "r") as f:
                self.difficult_words_list = f.read().split("\n")
            f.close()
        else:
            print("Error! Path: {} not found!".format(difficult_words_path))
        # Load stopwords list
        self.stopwords_list = []
        if stopwords_path == "":
            self.stopwords_list = set(stopwords.words('english'))
        elif os.path.exists(stopwords_path):
            with open(stopwords_path, "r") as f:
                self.stopwords_list = f.read().split("\n")
            f.close()
        else:
            print("Error! Path {} not found!".format(stopwords_path))
        # Set POS tags
        self.noun_tags = ["NN","NNS","NNP","NNPS"]
        self.verb_tags = ["MD","VB","VBD","VBG","VBN","VBP","VBZ"]
        self.adjective_tags = ["JJ", "JJR", "JJS"]
        self.adverbs_tags = ["RB","RBR","RBS","WRB"]
        # Set feature descriptions
        self.feature_descriptions = {
            "chars"            : "Character Count",
            "words"            : "Word Count",
            "4sqrt_words"      : "Fourth Root of Word Count",
            "avg_word_len"     : "Average Word Length",
            "words_gr5"        : "Word Count > 5 Char",
            "words_gr6"        : "Word Count > 6 Char",
            "words_gr7"        : "Word Count > 7 Char",
            "words_gr8"        : "Word Count > 8 Char",
            "diff_words"       : "Difficult Word Count",
            "long_words"       : "Long Word Count",
            "spell_err"        : "Spelling Errors",
            "uniq_words"       : "Unique Word Count",
            "nouns"            : "Noun Count",
            "verbs"            : "Verb Count",
            "adj"              : "Adjective Count",
            "adv"              : "Adverb Count",
            "stop_words"       : "Stop Words Count",
            "sentences"        : "Sentence Count",
            "avg_sentence_len" : "Average Sentence Length", 
            "exclamations"     : "Exclamation Mark Count",
            "questions"        : "Question Mark Count",
            "commas"           : "Comma Count"
        }

    def tokenize_sentences(self, id: int) -> list:
        """Tokenize essay by sentences"""
        if len(self.sentence_cache[id]) > 0:
            return self.sentence_cache[id]
        self.sentence_cache[id] = sent_tokenize(self.data.get_essay(id))
        return self.sentence_cache[id]

    def tokenize_words(self, id: int) -> list:
        """Tokenize essay by words"""
        if len(self.word_cache[id]) > 0:
            return self.word_cache[id]
        self.word_cache[id] = []
        text = self.data.get_essay(id)
        text = text.replace("/"," ")
        text = text.replace("-"," ")
        for word in word_tokenize(text):
            if "@" in word:
                word = word.lower()
            word_clean = re.sub(self.tokenize_filter, '', word)
            if len(word_clean) > 0:
                self.word_cache[id].append(word_clean)
        return self.word_cache[id]

    def get_pos_tags(self, id: int) -> int:
        """Get Part-Of-Speech tags of an essay"""
        if len(self.pos_tags_cache[id]) > 0:
            return self.pos_tags_cache[id]
        words = []
        for word in self.tokenize_words(id):
            if not word.lower() in self.stopwords_list:
                words.append(word)
        self.pos_tags_cache[id] = [tag for w, tag in pos_tag(words)]
        return self.pos_tags_cache[id]

    def characters(self, id: int) -> int:
        """Count characters in an essay"""
        return len(self.data.get_essay(id))

    def words(self, id: int) -> int:
        """Count words in an essay"""
        return len(self.tokenize_words(id))

    def fourth_root_of_word_count(self, id: int) -> float:
        """Calculate fourth root of word count in an essay"""
        return self.words(id) ** (1/4)

    def average_word_length(self, id: int) -> float:
        """Calculate average word length in an essay"""
        avg = 0
        words = self.tokenize_words(id)
        for word in words:
            avg += len(word)
        avg /= len(words)
        return avg

    def words_greater_n(self, id: int, n: int) -> int:
        """Count words that are longer than n characters in an essay"""
        g = 0
        for word in self.tokenize_words(id):
            if len(word) > n:
                g += 1
        return g

    def difficult_words(self, id: int) -> int:
        """Count difficult words in an essay"""
        count = 0
        for word in self.tokenize_words(id):
            if word in self.difficult_words_list:
                count += 1
        return count

    def long_words(self, id: int) -> int:
        """Count long words in an essay."""
        if self.total_average_word_length < 0:
            c = 0
            w = 0
            for i in range(self.data.count_essays()):
                c += self.characters(i)
                w += self.words(i)
            self.total_average_word_length = c/w
        count = 0
        for word in self.tokenize_words(id):
            if len(word) > self.total_average_word_length:
                count += 1
        return count

    def spelling_errors(self, id: int) -> int:
        """Count spelling errors in an essay"""
        count = 0
        for word in self.tokenize_words(id):
            if not self.spellcheck_dict.check(word) \
            and word != "th" \
            and word != "nt":
                count += 1
        return count

    def unique_words(self, id: int) -> int:
        """Count unique words in an essay"""
        unique = []
        for word in self.tokenize_words(id):
            word_lower = word.lower()
            if not word_lower in unique:
                unique.append(word_lower)
        return len(unique)
            
    def nouns(self, id: int) -> int :
        """Count nouns in an essay"""
        count = 0
        for tag in self.get_pos_tags(id):
            if tag in self.noun_tags:
                count += 1
        return count

    def verbs(self, id: int) -> int:
        """Count verbs in an essay"""
        count = 0
        for tag in self.get_pos_tags(id):
            if tag in self.verb_tags:
                count += 1
        return count

    def adjectives(self, id: int) -> int:
        """Count adjectives in an essay"""
        count = 0
        for tag in self.get_pos_tags(id):
            if tag in self.adjective_tags:
                count += 1
        return count

    def adverbs(self, id: int) -> int:
        """Count adverbs in an essay"""
        count = 0
        for tag in self.get_pos_tags(id):
            if tag in self.adverbs_tags:
                count += 1
        return count

    def stop_words(self, id: int) -> int:
        """Count stop words in an essay"""
        count = 0
        for word in self.tokenize_words(id):
            if word.lower() in self.stopwords_list:
                count += 1
        return count

    def sentences(self, id: int) -> int:
        """Count sentences in an essay"""
        return len(self.tokenize_sentences(id))

    def average_sentence_length(self, id: int) -> int:
        """Calculate average sentence length in an essay"""
        length = 0
        sentences = self.tokenize_sentences(id)
        for sentence in sentences:
            length += len(word_tokenize(re.sub(self.tokenize_filter, '', sentence)))
        length /= len(sentences)
        return length

    def exclamation_marks(self, id: int) -> int:
        """Count exclamation marks in an essay"""
        count = 0
        for c in self.data.get_essay(id):
            if c == "!":
                count += 1
        return count

    def question_marks(self, id: int) -> int:
        """Count question marks in an essay"""
        count = 0
        for c in self.data.get_essay(id):
            if c == "?":
                count += 1
        return count

    def commas(self, id: int) -> int:
        """Count commas in an essay"""
        count = 0
        for c in self.data.get_essay(id):
            if c == ",":
                count += 1
        return count

    def list_features(self):
        """List all available linguistic features"""
        for key in self.feature_descriptions:
            print("{}: {}".format(key, self.feature_descriptions[key]))

    def get_features(self, id: int, pretty = False) -> dict:
        """Get all features of the essay as a single object. If pretty is set to true, feature descriptions will be used as keys"""
        data = {
            "chars"            : self.characters(id),
            "words"            : self.words(id),
            "4sqrt_words"      : self.fourth_root_of_word_count(id),
            "avg_word_len"     : self.average_word_length(id),
            "words_gr5"        : self.words_greater_n(id, 5),
            "words_gr6"        : self.words_greater_n(id, 6),
            "words_gr7"        : self.words_greater_n(id, 7),
            "words_gr8"        : self.words_greater_n(id, 8),
            "diff_words"       : self.difficult_words(id),
            "long_words"       : self.long_words(id),
            "spell_err"        : self.spelling_errors(id),
            "uniq_words"       : self.unique_words(id),
            "nouns"            : self.nouns(id),
            "verbs"            : self.verbs(id),
            "adj"              : self.adjectives(id),
            "adv"              : self.adverbs(id),
            "stop_words"       : self.stop_words(id),
            "sentences"        : self.sentences(id),
            "avg_sentence_len" : self.average_sentence_length(id),
            "exclamations"     : self.exclamation_marks(id),
            "questions"        : self.question_marks(id),
            "commas"           : self.commas(id)
        }
        if not pretty:
            return data
        else:
            pretty_data = {}
            for key in data:
                pretty_data[self.feature_descriptions[key]] = data[key]
            return pretty_data
    
    def print_features(self, id: int):
        """Output essay features to command line"""
        print("Essay {}:".format(id))
        features = self.get_features(id, True)
        for key in features.keys():
            fmt = "\t{}: {}"
            if type(features[key]) is float:
                fmt = "\t{}: {:.2f}"
            print(fmt.format(key, features[key]))

    def generate_dataset(self, save_path: str = "", blacklist_features: list = [], column_names = True, normalize_scores = True):
        """Generate dataset and save it in a csv file. 
        Some features can be blacklisted (Optionally). 
        Column names can be disabled (Optionally).
        Score normalization can be disabled (Optionally)."""
        print("Generating linguistic features dataset")
        data = []
        # Add header
        if column_names:
            header = ["id","prompt","score"]
            for key in self.feature_descriptions.keys():
                if not key in blacklist_features:
                    header.append(key)
            data.append(header)
        # Add data
        for id in range(self.data.count_essays()):
            print("{}/{}".format(id+1, self.data.count_essays()))
            prompt = self.data.get_prompt(id)
            score = self.data.get_score_norm(id)
            features = self.get_features(id)
            item = [id,prompt,score]
            for key in features:
                if not key in blacklist_features:
                    item.append(features[key])
            data.append(item)
        # Save data
        if save_path != "":
            with open(save_path, "w") as f:
                for item in data:
                    f.write(",".join([str(value) for value in item])+"\n")
                f.close()
        return data

