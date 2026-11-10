# NLP Pipeline Introduction Workshop

**From raw text to machine-learning-ready features**  
**Course:** Foundations of Machine Learning — Conestoga College  
**Workshop:** Deconstructing Language for AI Agents  
**Instructor:** Prof. Espinosa  
**Project type:** Team workshop · Python / Jupyter Notebook

## Team members



| Member | Contribution |
|---|---|
| Ubaid Ullah | Workshop implementation |
| Ved Vishal Patel | execution, observations|
| Koushik Balne |  execution, observations |
| Sukhchain Singh |  repository updates (documented work)|

## Project overview

This workshop demonstrates how Natural Language Processing (NLP) converts unstructured human-language text into features that machine learning systems can use. It progresses from dataset inspection through tokenization, normalization, stopword removal, stemming, and word-frequency representations. It also explores a pretrained extractive question-answering model and reflects on practical trade-offs in preprocessing and LLM token usage.

The implementation follows the instructor's [NLP Pipeline Introduction Workshop](https://github.com/ProfEspinosaAIML/NLP_PipelineIntroductionWorkshop) and extends the challenge scaffolds with working code, numerical outputs, and written observations.

## Learning objectives

- Load and investigate a real-world text dataset.
- Implement and evaluate a reproducible text preprocessing pipeline.
- Compare stemming with lemmatization, including limitations of each.
- Represent processed text numerically with token-frequency counts (Bag-of-Words).
- Test factual question answering and identify limitations of an extractive model.
- Explore how word-level preprocessing affects illustrative token counts and costs, without confusing these with actual commercial LLM billing.

## Repository structure

```text
NLP_PipelineIntroductionWorkshop/
├── NLP-Pipeline-Introduction.ipynb  # Main executable workshop and four challenges
├── README.md                        # Project explanation and reproduction guide
├── requirements.txt                 # Instructor-provided Python dependencies
└── .gitignore                       # Excludes local and generated artifacts
```

The local `.venv/` environment and downloaded `nltk_data/` resources are intentionally excluded from Git.

## Dataset

The workshop uses scikit-learn's **20 Newsgroups** text dataset, loaded with:

```python
from sklearn.datasets import fetch_20newsgroups
newsgroups_data = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes")
)
```

**Observed initial exploration results:**

| Measure | Value |
|---|---:|
| Documents | 18,846 |
| Approximate total words | 3,423,145 |
| Initial unique vocabulary | 288,723 |

These totals reflect the workshop's text counting approach, not a linguistically standardized word count.

## Implementation and results

### 1. Core NLP preprocessing pipeline

The notebook combines the following stages:

```text
Raw document
    ↓
Regular-expression tokenization
    ↓
Lowercasing and stopword removal
    ↓
Porter stemming
    ↓
Cleaned token sequence
```

The original preprocessing experiment reported:

| Vocabulary metric | Observed result |
|---|---:|
| Initial unique words | 288,723 |
| Final unique words | 112,264 |
| Vocabulary reduction | 61.12% |

**Interpretation:** Preprocessing substantially reduced the distinct vocabulary. A smaller vocabulary can make some traditional NLP workflows more manageable, but reducing vocabulary does not automatically improve model accuracy or preserve all meaning.

### 2. Extractive question answering (DistilBERT)

The Hugging Face question-answering pipeline uses the default `distilbert/distilbert-base-cased-distilled-squad` model in the tested environment.

| Question | Extracted answer | Model confidence |
|---|---|---:|
| What does BERT stand for? | Bidirectional Encoder Representations from Transformers | 0.7050 |
| Who developed BERT? | Google | 0.9778 |
| Can BERT tell jokes? | Bidirectional Encoder Representations from Transformers | 0.0830 |
| Why is this joke funny? | I needed a break | 0.0999 |

**Interpretation:** The model extracted useful answers when the context contained the information. For questions requiring information not present in the context or explaining a joke, it returned unhelpful text spans. Confidence is not proof of correctness. This is an extractive QA model, not a general-purpose chatbot.

### 3. Challenge 1 — Preserve negation

The improved stopword filter retains `not`, `no`, `nor`, and `never` when they would otherwise be removed. In the tested sentence, the original pipeline removed `not` while the improved version preserved it. `never` was already present in both outputs.

**Learning:** Negation words can reverse meaning, particularly in sentiment analysis. A default stopword list should not be applied blindly.

### 4. Challenge 2 — Stemming versus lemmatization

| Original | Porter stem | WordNet lemma (default noun mode) |
|---|---|---|
| running | run | running |
| studies | studi | study |
| better | better | better |
| cars | car | car |
| wolves | wolv | wolf |
| playing | play | playing |

**Learning:** Stemming is simple and may produce non-dictionary forms; lemmatization often gives recognizable words but depends on part-of-speech assumptions. WordNet's default noun mode explains why some verb forms remained unchanged.

### 5. Challenge 3 — Bag-of-Words word frequencies

The notebook processes a **50-document subset** and counts the most common tokens. The five most frequent were:

| Token | Frequency |
|---|---:|
| `det` | 45 |
| `nyr` | 40 |
| `tor` | 39 |
| `bo` | 37 |
| `mtl` | 33 |

A five-term Bag-of-Words example for the first processed document yielded all zero counts. An additional example for document index `7` counted `bo` twice:

```python
{'det': 0, 'nyr': 0, 'tor': 0, 'bo': 2, 'mtl': 0}
```

**Learning:** Bag-of-Words maps text to numeric frequency features, but discards word order and much of the context. Frequently occurring abbreviations may also need domain-aware interpretation.

### 6. Challenge 4 — Token counts and illustrative prompt cost

The challenge uses a simple regular-expression **word tokenizer**, not a commercial model tokenizer. Its observed results were:

| Measure | Result |
|---|---:|
| Baseline word tokens | 114 |
| After stopword removal | 71 |
| After stemming | 71 |
| Word-token count reduction | 37.72% |
| Hypothetical rate | $0.008 per 1,000 word tokens |
| Hypothetical baseline cost | $0.000912 |
| Hypothetical processed cost | $0.000568 |
| Hypothetical savings | $0.000344 |

**Learning:** Stopword removal reduced the word-token count by 43, whereas stemming changed word forms without reducing the number of items. The costs are illustrative classroom calculations **not provider prices or actual API bills**. Real LLM tokenization, prompt and completion lengths, and price tiers determine actual costs. Removing context may harm answer quality.

## Setup and reproduction

**Tested local environment:** Windows, Python **3.13.7**, scikit-learn **1.7.0**, NLTK **3.9.1**, Matplotlib **3.10.3**, Transformers **4.53.0**, and PyTorch **2.7.1+cpu**. The instructor notebook records a different Python version; other compatible versions may work but have not been verified in this repository.

1. Clone the team repository:

   ```powershell
   git clone https://github.com/UbaidUllah777/NLP_PipelineIntroductionWorkshop.git
   cd NLP_PipelineIntroductionWorkshop
   ```

2. Create and activate a virtual environment (PowerShell):

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the provided dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

   **Windows note:** The instructor's pinned dependencies include Windows-oriented packages; installation on a different operating system may require adjusting the environment.

4. Open `NLP-Pipeline-Introduction.ipynb` in VS Code or Jupyter and select the project's `.venv` kernel. If the kernel is not available, install or register `ipykernel` in that environment.

5. Run the notebook **from the first cell to the last**. Its setup cell downloads NLTK resources such as `punkt`, `punkt_tab`, `stopwords`, and `wordnet` to the project's local `nltk_data/` directory. The notebook also downloads the 20 Newsgroups dataset and may download the pretrained question-answering model on its first run. Internet access is therefore needed for a fresh setup.

6. Confirm the dataset statistics, pipeline transformation, QA experiments, all four challenges, and the associated Markdown observations appear without unhandled execution errors. Informational Hugging Face cache, CPU, and Jupyter widget warnings may appear without preventing execution.

## Validation checklist

- [x] Environment imports and Jupyter kernel checked locally
- [x] Dataset loaded and examined
- [x] Tokenization, normalization, stopword removal, and stemming executed
- [x] Full preprocessing pipeline and vocabulary reduction reported
- [x] Extractive QA questions, output confidence scores, and limitations documented
- [x] Four challenge implementations and their talking points completed
- [x] Final notebook restarted and run through (as recorded in project history)
- [ ] Confirm final team roster, individual contributions, and submission instructions
- [ ] Complete instructor rubric audit before submission

## Limitations and future improvements

- Stopword removal may discard context, especially negation and grammatical relations.
- Stemming may produce non-dictionary word forms; lemmatization depends on part-of-speech information.
- Bag-of-Words loses sequence and context and is not equivalent to modern embeddings.
- Extractive QA can return irrelevant spans instead of declining to answer.
- Word-token cost estimates should not be used as real commercial LLM billing predictions.
- The `requirements.txt` file was inherited from the instructor; a future cross-platform release could use more flexible dependency constraints and an explicit model identifier/revision.

## Source and acknowledgments

Workshop scaffold and learning materials: [ProfEspinosaAIML / NLP_PipelineIntroductionWorkshop](https://github.com/ProfEspinosaAIML/NLP_PipelineIntroductionWorkshop). The team's code completions, experimental output, and interpretations are recorded in its own [GitHub repository](https://github.com/UbaidUllah777/NLP_PipelineIntroductionWorkshop).
