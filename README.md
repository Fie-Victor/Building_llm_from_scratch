# Project building a LLM From Scratch

In this project (inpired from https://github.com/rasbt/LLMs-from-scratch), I learn to build and use :

- An Byte Pair Encoding (BPE) tokenizer
- An embedding encoder
- the transformer architechture
- the attention mechanism
- a GPT-2 Like LLM pretrained
- the finetuning of a llm for classification and instruction-following
- the Low rank Adaptation (LORA) finetuning method.


The code is organise in chapter containing notebooks to go throught the implementation step by step while reading this book: 
[Build a LLM from Scratch - Sebastian Raschka](https://www.manning.com/books/build-a-large-language-model-from-scratch)

(optional) Before running the codes : 

```python
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\activate for Windows CMD or .\venv\Scripts\Activate.ps1 for Windows Powershell)
pip install -r requirements.txt
```

NB: You can find all utils fonction and implementation in utils.py

NB: we use the file gpt_downloader.py to dowload the file gpt_download.py
and then we use gpt_download.py to download and load openai gpt2 publicly available weigts from their blob storage
