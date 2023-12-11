# AES Tools

Abstraction layers for simplifying experiments with AES

## Documentation
You can use `pydoc` to read documentation for any class. E.g.:

```bash
pydoc AESData
pydoc AESEmbeddings
```

## Classes

### Dataset

```python
from AESData import AESData
data = AESData("datasets/your-dataset.json")
print(data.get_essay(5))
```

Provides abstraction for using various datasets for AES.
For each dataset, a special JSON manifest file is required:

```json
{   
    "id": "UNIQUE_DATASET_ID",
    "path": "PATH_TO_DATASET",
    "author": ["AUTHOR1", "AUTHOR2", ...],
    "title": "DATASET_NAME",
    "publisher": "PUBLISHER_NAME",
    "year": YEAR_OF_PUBLICATION,
    "url": "https://url-where-it-can-be-downloaded",
    "skip_first_line": true, /* Whether the first line will be skipped. */
    "columns": { /* Indexes of columns where relevant data is located. */
        "prompt": i1,
        "essay": i2,
        "score": i3
    },
    "prompts": [ /* Metadata for each prompt. */
        {
            "id": 1, /* Index (just for aesthetics). */
            "min_score": S_MIN, /* Min and Max allowed scores. */
            "max_score": S_MAX,
            "genre": "Argument" /* Genre of the essay. */
        },
        ...
    ],
    "special_tokens": ["@ORGANIZATION", "@LOCATION", ...], /* List of special tokens that mask certain words in dataset */
    "alternative_tokens": ["organization", "location", ...] /* Replacements for special tokens (may improve results)*/
}
```

### Embeddings

```python
from AESEmbeddings import AESEmbeddings, AESSentenceEmbeddings
embeddings = AESEmbeddings(data) # Requires AESData object
print(embeddings.get_embeddings(5))
```

Helps to generate and cache embeddings for any essay by its id.

`AESEmbeddings` uses standard BERT models from `transformers` library. Model can be specified using `model_name` constructer parameter.

`AESSentenceEmbeddings` uses `sentence_transformers` library. Model can be specified using `model_name` constructer parameter.

### Linguistic Features

```python
from AESLinguisticFeatures import AESLinguisticFeatures
features = AESLinguisticFeatures(data)
print(features.spelling_errors(5))
```

`AESLinguisticFeatures.py` contains `AESLinguisticFeatures` class which helps to extract linguistic features from essays. Refer to `pydoc` to get more information. Currently the following features are supported:

| Feature name     | Description               |
|------------------|---------------------------|
| chars            | Character Count           |
| words            | Word Count                |
| 4sqrt_words      | Fourth Root of Word Count |
| avg_word_len     | Average Word Length       |
| words_gr5        | Word Count > 5 Char       |
| words_gr6        | Word Count > 6 Char       |
| words_gr7        | Word Count > 7 Char       |
| words_gr8        | Word Count > 8 Char       |
| diff_words       | Difficult Word Count      |
| long_words       | Long Word Count           |
| spell_err        | Spelling Errors           |
| uniq_words       | Unique Word Count         |
| nouns            | Noun Count                |
| verbs            | Verb Count                |
| adj              | Adjective Count           |
| adv              | Adverb Count              |
| stop_words       | Stop Words Count          |
| sentences        | Sentence Count            |
| avg_sentence_len | Average Sentence Length   |
| exclamations     | Exclamation Mark Count    |
| questions        | Question Mark Count       |
| commas           | Comma Count               |

> Eid, S.M. and Nayer Wanas (2017). Automated essay scoring linguistic feature: Comparative study. doi:https://doi.org/10.1109/accs-peit.2017.8303043.

> Murray, K. and Orii, N. (n.d.). Automatic Essay Scoring. [online] Available at: http://www.cs.cmu.edu/~norii/pub/aes.pdf

> Larkey, L.S. (1998). Automatic essay grading using text categorization techniques. doi:https://doi.org/10.1145/290941.290965. 