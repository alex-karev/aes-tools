# Variaous Experiments with AES

## Dataset Abstraction

`AESData.py` has a special `AESData` class which provides an abstraction for using various AES datasets.
For each dataset, a special JSON card is required:

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
    ]
}
```

Then the dataset can be easily loaded from code:

```python
from AESData import AESData
data = AESData("datasets/your-dataset.json")
```

You can use `pydoc` to read the class documentation:

```bash
pydoc AESData
```

## Embeddings Abstraction

`AESEmbeddings.py` contains `AESEmbeddings` class which makes it possible to generate and cache embeddings for any essay by id.

You can use initialize `AESEmdeddigns` with `AESData` as argument and then use `get_embeddings` function to generate embeddings for any essay.