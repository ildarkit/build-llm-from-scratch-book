import torch
import tiktoken
from ch04.ex_4_1_params_count import GPTModel
from ch05.gpt_download import download_and_load_gpt2
from ch05.ex_5_6_gpt_compare import GPT_CONFIG_124M, load_weights_into_gpt
from ch05.ex_5_2_ktop_temperature import calc_loss_loader, create_dataloader_v1


if __name__ == '__main__':
    context_length = 256

    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    tokenizer = tiktoken.get_encoding("gpt2")

    train_loader = create_dataloader_v1(
        train_data,
        tokenizer,
        batch_size=2,
        max_length=context_length,
        stride=context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        val_data,
        tokenizer,
        batch_size=2,
        max_length=context_length,
        stride=context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt = GPTModel(GPT_CONFIG_124M)
    gpt.to(device)
    gpt.eval()

    _settings, params = download_and_load_gpt2(
        model_size="124M", models_dir="gpt2"
    )
    load_weights_into_gpt(gpt, params)

    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, gpt, device)
        val_loss = calc_loss_loader(val_loader, gpt, device)

    print(f"Потери на обучающей выборке (Train Loss): {train_loss:.4f}")
    print(f"Потери на проверочной выборке (Val Loss): {val_loss:.4f}")

