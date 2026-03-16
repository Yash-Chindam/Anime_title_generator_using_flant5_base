# Anime Title Generator using FLAN-T5 Base

An NLP project that generates creative and contextual anime titles using the FLAN-T5 base language model, leveraging transfer learning for creative text generation.

## 📋 Overview

This project implements an anime title generation system using the FLAN-T5 model. It takes anime plot descriptions, characters, genres, or other attributes as input and generates relevant, creative, and context-aware anime titles using state-of-the-art transformer-based models.

## 🎯 Objectives

- Generate creative anime titles from plot descriptions
- Leverage FLAN-T5 base model capabilities
- Demonstrate fine-tuning on domain-specific data
- Create contextually relevant titles
- Handle various input formats and styles
- Evaluate title quality and relevance

## 🗂️ Project Structure

```
Anime_title_generator_using_flant5_base/
├── Anime_title_generator_using_flant5_base.ipynb  # Main notebook
├── README.md                                      # Project documentation
└── data/                                          # Dataset (if available)
```

## 🛠️ Technologies & Libraries

- **Transformer Model**: FLAN-T5 (Hugging Face)
- **Deep Learning**: PyTorch, TensorFlow
- **NLP Toolkit**: Hugging Face Transformers, Tokenizers
- **Data Processing**: Pandas, NumPy
- **Evaluation**: BLEU, ROUGE, CIDEr, Human evaluation
- **Utilities**: tqdm, collections

## 📊 Key Features

- **FLAN-T5 Integration**:
  - Pre-trained instruction-following model
  - Fine-tuning capabilities
  - Zero-shot and few-shot learning
  - Multi-task support

- **Title Generation**:
  - Single-word and multi-word titles
  - Contextual understanding
  - Creative variations
  - Style-specific generation

- **Input Types**:
  - Plot summaries
  - Character descriptions
  - Genre specifications
  - Anime themes
  - Mixed attributes

- **Generation Methods**:
  - Greedy decoding
  - Beam search
  - Temperature sampling
  - Top-k and nucleus sampling

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- GPU recommended (6GB+ VRAM)
- Internet connection (for model downloads)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Anime_title_generator_using_flant5_base
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download FLAN-T5 model:
   ```bash
   python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
   AutoTokenizer.from_pretrained('google/flan-t5-base'); \
   AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')"
   ```

4. Open the notebook:
   ```bash
   jupyter notebook Anime_title_generator_using_flant5_base.ipynb
   ```

## 📈 Model Architecture

```
Input Plot/Description
         ↓
   Tokenization
         ↓
   FLAN-T5 Encoder
         ↓
   Sequence to Sequence
         ↓
   FLAN-T5 Decoder
         ↓
   Title Generation
         ↓
  Post-processing (Optional)
```

## 💾 Training Configuration

```python
MODEL_NAME = "google/flan-t5-base"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 3
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
SEED = 42
```

## 📝 Usage Example

### Basic Title Generation

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Example plot description
plot = "A young protagonist discovers hidden powers and fights against an evil empire."

# Prepare input
input_ids = tokenizer(
    f"Generate anime title: {plot}",
    max_length=512,
    truncation=True,
    return_tensors="pt"
).input_ids

# Generate title
title_ids = model.generate(input_ids, max_length=50, num_beams=4)
title = tokenizer.decode(title_ids[0], skip_special_tokens=True)

print(f"Generated Title: {title}")
```

### Batch Generation

```python
plot_descriptions = [
    "A student discovers he's the chosen one.",
    "Magical girls fight monsters to save the world.",
    "Childhood friends reunite in a magical forest."
]

generated_titles = []
for plot in plot_descriptions:
    input_ids = tokenizer(
        f"Generate anime title: {plot}",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).input_ids
    
    title_ids = model.generate(input_ids, temperature=0.7, top_p=0.9)
    title = tokenizer.decode(title_ids[0], skip_special_tokens=True)
    generated_titles.append(title)

for desc, title in zip(plot_descriptions, generated_titles):
    print(f"Plot: {desc}")
    print(f"Generated Title: {title}\n")
```

### Fine-tuning on Custom Data

```python
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments

class AnimeDataset(Dataset):
    def __init__(self, descriptions, titles, tokenizer, max_length=512):
        self.descriptions = descriptions
        self.titles = titles
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, idx):
        description = self.descriptions[idx]
        title = self.titles[idx]
        
        input_encoding = self.tokenizer(
            f"Generate anime title: {description}",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            title,
            max_length=50,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

# Prepare training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_steps=100,
    save_steps=500
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
```

## ⚙️ Hyperparameters

```python
TEMPERATURE = 0.7          # Sampling diversity
TOP_K = 50                 # Top-k sampling
TOP_P = 0.95               # Nucleus sampling
NUM_BEAMS = 4              # Beam search width
EARLY_STOPPING = True
LENGTH_PENALTY = 2.0       # Encourage longer sequences
REPETITION_PENALTY = 2.0   # Avoid repetition
```

## 📊 Evaluation Metrics

- **BLEU Score**: Measures n-gram overlap with reference titles
- **ROUGE Score**: Evaluates summary-like quality
- **CIDEr**: Consensus-based image description evaluation
- **Semantic Similarity**: Using sentence transformers
- **Human Evaluation**:
  - Relevance (1-5)
  - Creativity (1-5)
  - Appropriateness (1-5)

## 🔍 Example Outputs

| Plot/Attribute | Generated Title |
|---|---|
| "Student discovers magic" | "Academy of Awakening Powers" |
| "Time-traveling adventure" | "Chronicles of the Temporal Gate" |
| "Childhood friends reunion" | "Bonds Beyond Time" |
| "Battle against darkness" | "Light Against the Shadows" |

## 🎯 Advanced Techniques

- **Few-Shot Prompting**: Provide examples for better results
- **Prompt Engineering**: Optimize input prompts
- **Ensemble Methods**: Combine multiple generation runs
- **Post-processing**: Filter and rank generated titles
- **Fine-tuning**: Adapt model to anime domain

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## 📚 References

- [FLAN-T5: Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416)
- [Exploring the Limits of Transfer Learning with a Unified Transformer](https://arxiv.org/abs/1910.10683)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## 📄 License

This project is open source and available under the MIT License.

## 💡 Best Practices

- Cache model after first load
- Use batch processing for efficiency
- Experiment with different prompts
- Validate generated titles with domain experts
- Monitor model output for biases
- Store generation parameters for reproducibility

## ✉️ Contact

For questions or suggestions, please open an issue or contact the project maintainers.