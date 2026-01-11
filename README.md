# LLaMA 3.1-8B Bengali Empathetic Conversations Fine-tuning

## Project Overview
Fine-tuned LLaMA 3.1-8B-Instruct on Bengali Empathetic Conversations dataset using LoRA for parameter-efficient training on Kaggle's free GPU.

## Quick Start

### Files Included
- `llama_bengali_finetuning.ipynb` - Complete training notebook
- `llama_bengali/` - Fine-tuned model (LoRA adapters)
- `llama_experiments.db` - Training logs and metrics
- `training_report.txt` - Detailed documentation
- `evaluation_results.txt` - Evaluation metrics and analysis

### Key Results
- **Final Training Loss:** 0.6059
- **Final Validation Loss:** 0.4941
- **Training Loss Progression:**
  - Step 250: 0.5479
  - Step 500: 0.5060 (Validation: 0.5148)
  - Final: 0.4941
- **ROUGE-1:** 0.0954
- **ROUGE-2:** 0.0078
- **ROUGE-L:** 0.0819
- **BLEU:** 6.45
- **Perplexity:** 1.70

### Model Architecture
- **Base Model:** meta-llama/Meta-Llama-3.1-8B-Instruct
- **Method:** LoRA (r=8, alpha=16)
- **Quantization:** 4-bit (NF4)
- **Target Modules:** q_proj, v_proj
- **Trainable Parameters:** 3,407,872 (~0.07% of total)

### Training Configuration
- **Dataset:** 2,500 samples (Bengali Empathetic Conversations)
  - Training: 2,250 samples
  - Validation: 250 samples
- **Max Sequence Length:** 2048 tokens (covers >99% of data)
- **Training Steps:** 500
- **Evaluation Frequency:** Every 250 steps
- **Save Frequency:** Every 250 steps
- **Batch Size:** 1 (effective 8 with gradient accumulation)
- **GPU:** Kaggle T4 x2 (15GB)
- **Training Time:** ~9 hours 6 minutes
  - Start: 16:43:49
  - End: 01:49:56

### Design Patterns Implemented
- **Strategy Pattern:** Swappable LoRA configurations
- **Factory Pattern:** Dataset processing pipeline
- **Repository Pattern:** Database logging

### How to Use the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./llama_bengali")
tokenizer = AutoTokenizer.from_pretrained("./llama_bengali")

# Set model to evaluation mode
model.eval()

# Generate response
def generate_empathetic_response(question, max_length=150):
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n"
        f"তুমি সহানুভূতিশীল পরামর্শদাতা।<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\\n\\n"
        f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split('assistant')[-1].strip()
    
    return response

# Example usage
question = "আমি খুব দুশ্চিন্তায় আছি।"
response = generate_empathetic_response(question)
print(response)
```

### Key Challenges Solved
1. ✅ Protobuf compatibility issues in Kaggle environment
2. ✅ GPU memory optimization (15GB limit)
3. ✅ Sequence length analysis and justification (2048 tokens based on data distribution)
4. ✅ Manual ROUGE/BLEU calculation (no library dependencies)
5. ✅ Human evaluation pipeline implementation
6. ✅ Extended training time management within Kaggle's 12-hour limit

### Dataset Statistics
- **Total Loaded:** 38,233 conversation pairs
- **After Cleaning:** 22,609 valid pairs
- **Sampled for Training:** 2,500 samples
- **Sequence Length Analysis:**
  - Median: 144 tokens
  - 90th percentile: 249 tokens
  - 95th percentile: 342 tokens
  - 99th percentile: 847 tokens
  - Coverage at 2048: >99%

### Memory Optimization Techniques
1. 4-bit NF4 Quantization (~75% memory reduction)
2. LoRA with rank 8 (0.07% trainable parameters)
3. Gradient checkpointing
4. Limited target modules (q_proj, v_proj only)
5. Paged AdamW 8-bit optimizer
6. Mixed precision training (FP16)

### Files Description
- **1_CODE/**: Complete notebook with all preprocessing, training, and evaluation
- **2_MODEL/**: LoRA adapter weights and tokenizer
- **3_EVALUATION/**: Metrics, database logs, human eval samples
- **4_DOCUMENTATION/**: Training report, challenges, decisions
- **5_SAMPLE_OUTPUTS/**: Example model responses

## Author
**Tanvir Ahammed**  
📧 Email: tanvir7535@gmail.com