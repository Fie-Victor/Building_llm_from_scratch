# Project building a LLM From Scratch

In this project (inpired from https://github.com/rasbt/LLMs-from-scratch), I learn to build and use :

- An Byte Pair Encoding (BPE) tokenizer
- An embedding encoder
- the transformer architechture
- the attention mechanism
- the finetuning of a llm for classification and instruction-following
- the Low rank Adaptation (LORA) finetuning method.

Before running the codes:

```python
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate for Windows CMD or .\venv\Scripts\Activate.ps1 for Windows Powershell)
pip install -r requirements.txt
```

NB: You can find all utils fonction and implementation in utils.py
NB: we use the file gpt_downloader.py to dowload the file gpt_download.py
and then we use gpt_download.py to download and load openai gpt2 public ly available weigts
