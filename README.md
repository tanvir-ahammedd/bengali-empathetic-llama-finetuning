\# LLaMA 3.1-8B Bengali Empathetic Conversations Fine-tuning



\## Project Overview

Fine-tuned LLaMA 3.1-8B-Instruct on Bengali Empathetic Conversations dataset using LoRA for parameter-efficient training on Kaggle's free GPU.



\## Quick Start



\### Files Included

\- `llama\_bengali\_finetuning.ipynb` - Complete training notebook

\- `llama\_bengali/` - Fine-tuned model (LoRA adapters)

\- `llama\_experiments.db` - Training logs and metrics

\- `training\_report.txt` - Detailed documentation

\- `evaluation\_results.txt` - Evaluation metrics and analysis



\### Key Results

\- \*\*Training Loss:\*\* 0.5403

\- \*\*ROUGE-1:\*\* 0.1108

\- \*\*ROUGE-2:\*\* 0.0044

\- \*\*ROUGE-L:\*\* 0.0882

\- \*\*BLEU:\*\* 3.67

\- \*\*Perplexity:\*\* \[CHECK YOUR VALUE]



\### Model Architecture

\- \*\*Base Model:\*\* meta-llama/Meta-Llama-3.1-8B-Instruct

\- \*\*Method:\*\* LoRA (r=8, alpha=16)

\- \*\*Quantization:\*\* 4-bit (NF4)

\- \*\*Target Modules:\*\* q\_proj, v\_proj

\- \*\*Trainable Parameters:\*\* ~0.1% of total



\### Training Configuration

\- \*\*Dataset:\*\* 4,000 samples (Bengali Empathetic Conversations)

\- \*\*Max Sequence Length:\*\* 512 tokens (covers 93.4% of data)

\- \*\*Training Steps:\*\* 1,000

\- \*\*Batch Size:\*\* 1 (effective 8 with gradient accumulation)

\- \*\*GPU:\*\* Kaggle T4 x2 (15GB)

\- \*\*Training Time:\*\* ~2-3 hours



\### Design Patterns Implemented

\- \*\*Strategy Pattern:\*\* Swappable LoRA configurations

\- \*\*Factory Pattern:\*\* Dataset processing pipeline

\- \*\*Repository Pattern:\*\* Database logging



\### How to Use the Model

```python

from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel



\# Load base model

base\_model = AutoModelForCausalLM.from\_pretrained(

&nbsp;   "meta-llama/Meta-Llama-3.1-8B-Instruct"

)



\# Load LoRA adapters

model = PeftModel.from\_pretrained(base\_model, "./llama\_bengali")

tokenizer = AutoTokenizer.from\_pretrained("./llama\_bengali")



\# Generate response

prompt = "আমি খুব দুশ্চিন্তায় আছি।"

inputs = tokenizer(prompt, return\_tensors="pt")

outputs = model.generate(\*\*inputs, max\_new\_tokens=150)

response = tokenizer.decode(outputs\[0], skip\_special\_tokens=True)

```



\### Key Challenges Solved

1\. ✅ Protobuf compatibility issues in Kaggle environment

2\. ✅ GPU memory optimization (15GB limit)

3\. ✅ Sequence length analysis and justification

4\. ✅ Manual ROUGE/BLEU calculation (no library dependencies)

5\. ✅ Human evaluation pipeline implementation



\### Files Description

\- \*\*1\_CODE/\*\*: Complete notebook with all preprocessing, training, and evaluation

\- \*\*2\_MODEL/\*\*: LoRA adapter weights and tokenizer

\- \*\*3\_EVALUATION/\*\*: Metrics, database logs, human eval samples

\- \*\*4\_DOCUMENTATION/\*\*: Training report, challenges, decisions

\- \*\*5\_SAMPLE\_OUTPUTS/\*\*: Example model responses



\## Author



\*\*Tanvir Ahammed\*\*  

📧 Email: tanvir7535@gmail.com

