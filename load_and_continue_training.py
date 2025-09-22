"""
File to test loading a pretrained model and continue training it on a custom dataset."""

import torch
import tiktoken

from utils import (
    create_dataloader_v1,
    GPT_CONFIG_124M,
    generate, text_to_token_ids,
    token_ids_to_text,
    train_model_simple,
    GPTModel
    )


file_path = 'data_loading_and_tokenization/the-verdict.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    text_data = f.read()

# Next, we divide the dataset into a training and a validation set and use the data loaders
# from chapter 2 to prepare the batches for LLM trainin

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = tiktoken.get_encoding("gpt2")
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)




if __name__ =="__main__":

    print("Loading pretrained model and optimiser...")
    checkpoint = torch.load("pretraining_on_unlabelled_data/model_and_optimiser.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()

    print("Startint continuation training...")
    _, _, _ = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=1,
        device=device,
        eval_freq = 5,
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    print("End of training, generating text...")
    torch.manual_seed(123)

    model.eval()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you ", tokenizer).to(device),
        max_new_tokens=15,
        context_size= GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=0.7
    )
    print("\n\n")
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))