import torch
from ch04.ex_4_1_params_count import GPTModel


VOCAB = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

INVERSE_VOCAB = {v: k for k, v in VOCAB.items()}


def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
          for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {INVERSE_VOCAB[i]}")


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


if __name__ == "__main__":
    GPT_CONFIG = {
       "emb_dim": 768,
       "n_heads": 12,
       "n_layers": 12,
       "context_length": 4,
       "batch_size": 1,
       "drop_rate_att": 0.1,
       "drop_rate_emb": 0.3,
       "drop_rate_shortcut": 0.3,
       "vocab_size": len(VOCAB),
       "qkv_bias": False,
    }

    model = GPTModel(GPT_CONFIG)  

    start_context = "every effort moves you"
    encoded = [VOCAB[word] for word in start_context.split()]
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()
    with torch.no_grad():
        logits = model(encoded_tensor)
    next_token_logits = logits[0, -1, :]

    temperatures = [1, 0.1, 5]
    print("\nsampling frequencies of probabilities softmax scaled by temperature: "
        f"{temperatures}")
    for t in temperatures:
        probas = softmax_with_temperature(next_token_logits, t)
        print(f"temperature {t}")
        print_sampled_tokens(probas)

