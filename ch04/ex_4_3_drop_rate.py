import torch
import tiktoken
from ch04.ex_4_1_params_count import GPTModel


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


if __name__ == "__main__":
    GPT_CONFIG_124M = {
       "emb_dim": 768,
       "n_heads": 12,
       "n_layers": 12,
       "context_length": 1024,
       "batch_size": 2,
       "drop_rate_att": 0.3,
       "drop_rate_emb": 0.1,
       "drop_rate_shortcut": 0.2,
       "vocab_size": 50257,
       "qkv_bias": False,
    }
    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

