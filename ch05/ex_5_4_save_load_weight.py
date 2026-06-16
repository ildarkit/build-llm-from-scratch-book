import torch
import tiktoken
from ch04.ex_4_1_params_count import GPTModel
from ch05.ex_5_2_ktop_temperature import (
    text_to_token_ids,
    token_ids_to_text,
    train_model_simple,
    generate,
    create_dataloader_v1
)


if __name__ == '__main__':
    train_ratio = 0.90
    num_epochs = 1
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = "the-verdict.txt"
    model_weight_file = "model_and_optimizer.pth"
    start_context="Every effort moves you"

    GPT_CONFIG_124M = {
       "emb_dim": 768,
       "n_heads": 12,
       "n_layers": 12,
       "context_length": 256,
       "batch_size": 2,
       "drop_rate_att": 0.1,
       "drop_rate_emb": 0.2,
       "drop_rate_shortcut": 0.3,
       "vocab_size": 50257,
       "qkv_bias": False,
    }

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    checkpoint = torch.load(model_weight_file, map_location=device)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    tokenizer = tiktoken.get_encoding("gpt2")
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=start_context, tokenizer=tokenizer
    )

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=5,
        temperature=0.5
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

