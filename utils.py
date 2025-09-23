"""
Module pour garder les fonctions utilistaires
"""

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids =[]
        self.target_ids = []

        #tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        #we use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range (0, len(token_ids)- max_length, stride):
            input_chunk = token_ids[i: i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers=0):

    #initializing the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    #we create the dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    #we create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads                                              #A
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)    
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)                                         #B
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal = 1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape                                               
        keys = self.W_key(x)
        queries = self.W_query(x)                                                       #C
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)                 #D
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1,2) # (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1,2)                                                  #E
        queries = queries.transpose(1,2)


        attn_scores = queries@keys.transpose(2,3)                                       #F
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]                          #G

        attn_scores.masked_fill_(mask_bool, -torch.inf)                                 #H         

        attn_weigths = torch.softmax(attn_scores /keys.shape[-1]**0.5, dim=-1)
        attn_weigths = self.dropout(attn_weigths)

        context_vec = (attn_weigths@values).transpose(1,2)                              #I

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)          #J
        context_vec = self.out_proj(context_vec)                                         #K
        return context_vec
#now we implement GELU (Gaussian Error Linear Unit) activation fonction very popular in transformer models
#GELU(x) = x * P(X <= x) where X ~ N(0,1) GELU(x) ~ x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))* (x+0.044715* torch.pow(x,3))))
    

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) #trainable parameter to learn correctly scaling
        self.shift = nn.Parameter(torch.zeros(emb_dim)) #trainable parameter to learn correctly scaling

    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x-mean) / torch.sqrt(var + self.eps) #to prevent division by 0
        return self.scale * x_norm + self.shift
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]), #increase dimension by 4 times
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]) #project back to original dimension
        )

    def forward(self, x):
        return self.layers(x)
    

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])


    def forward(self,x):
                                                                                #A
        shortcut =x
        x = self.norm1(x)
        x= self.att(x)
        x= self.drop_shortcut(x)
        x = x + shortcut #add the original input (shortcut connection)

        shortcut = x
        x  = self.norm2(x)                                                      #B
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut                                                         #C                                             
        return x

#A Shortcut connection for attention block
#B Shortcut connection for feed forward block
#C Add the original input back



#The GPT Model architecture implementation
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device)) #A

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
    
#A The device setting will allow us to train the model on a CPU or GPU, depending on which device the input data sits


#transform the output fom GPT in text
#A function for the GPT model to generate text

def generate_text_simple(model, idx, max_new_tokens, context_size): #A
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]                           #B
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]                                  #C
        probas = torch.softmax(logits, dim=-1)                       #D
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  #E
        idx = torch.cat((idx, idx_next), dim=1)                   #F

    return idx

#A idx is a (batch, n_tokens) array of indices in the current context
#B Crop current context if it exceeds the supported context size E.g., if LLM supports only 5 tokens, and the
#context size is 10 then only the last 5 tokens are used as context
#C Focus only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
#D probas has shape (batch, vocab_size)
#E idx_next has shape (batch, 1)
#F Append sampled index to the running sequence, where idx has shape (batch, n_tokens+1)


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,   #A
    "emb_dim": 768,       
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,           #B
    "qkv_bias": False,
}

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) #remove batch dimension
    return tokenizer.decode(flat.tolist())

# we implement a utility function to calculate the cross entropy loss of a given batch
# returned via the training and validation loader:

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device) #A
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


#A the transfer to a given device allows us to transfer the data to a GPU

#Function to compute the training and validation loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)                                  #A
    else:
        num_batches = min(num_batches, len(data_loader))                #B

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()                                    #C
        else:
            break

    return total_loss / num_batches                                       #D



#A Iterative over all batches if no fixed num_batches is specified
#B Reduce the number of batches to match the total number of batches in the data loader if num_batches
#exceeds the number of batches in the data loader
#C Sum loss for each batch
#D Average the loss over all batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() #A
    with torch.no_grad(): #B
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

#A Dropout is disabled during evaluation for stable, reproducible results
#B Disable gradient tracking, which is not required during evaluation, to reduce the computational overhead

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
        model=model, idx=encoded,
        max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " ")) # Compact print format
    model.train()


#simple training function
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []  #A
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):                       #B
        model.train()  #training mode
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                        #C
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                              #D
            optimizer.step()                             #E
            tokens_seen += input_batch.numel()
            global_step +=1


            if global_step %eval_freq  ==0 :             #F
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}):" 
                      f" Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")
                
        generate_and_print_sample(model, tokenizer,device, start_context) #G

    return train_losses, val_losses, track_tokens_seen

#A Initialize lists to track losses and tokens seen
#B Start the main training loop
#C Reset loss gradients from previous batch iteration
#D Calculate loss gradients
#E Update model weights using loss gradients
#F Optional evaluation step
#G Print a sample text after each epoch

#A modified text generation function with more diversity
def generate(model, idx, max_new_tokens, context_size, temperature =1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):                         #A
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # just the probas for last token
        if top_k is not None:                               #B
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
            logits
            )

        if temperature > 0.0:                                  #C
            logits = logits / temperature     
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:                                                  #D
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
    
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

#A For-loop is the same as before: Get logits, and only focus on last time step
#B In this new section, we filter logits with top_k sampling
#C This is the new section where we apply temperature scaling
#D Carry out greedy next-token selection as before when temperature scaling is disabled
#E Stop generating early if end-of-sequence token is encountered and eos_id is specified

#LOADING oPENaI WEIGHTS iNTO OUR gpt MODEL CODE

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt,params):

    import numpy as np
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])  #A
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params['blocks'])):                       #B
        q_w, k_w, v_w = np.split(                                #C
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)


        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
        
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"]) #D

#A Setting the model's positional and token embedding weights to those specified in params.
#B Iterate over each transformer block in the model.
#C The np.split function is used to divide the attention and bias weights into three equal parts for the query,
#key, and value components.
#D The original GPT-2 model by OpenAI reused the token embedding weights in the output layer to reduce the
#total number of parameters, which is a concept known as weight tying.
    
    
if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12, # numbers of head in multiattention head
        "n_layers":12, #number of transformer bloclks
        "drop_rate": 0.1,
        "qkv_bias": False, #"first we don't add bias to linear layers for keys, value and queries parameters"
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total number of parameters: {total_params:,}")

