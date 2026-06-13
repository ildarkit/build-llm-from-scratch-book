from .ex_4_1_params_count import GPTModel


def count_parameters(model_module):
    return sum(p.numel() for p in model_module.parameters() if p.requires_grad)


if __name__ == "__main__":
    GPT_CONFIG_MED = {
       "emb_dim": 1024,
       "n_heads": 16,
       "n_layers": 24,
       "context_length": 1024,
       "batch_size": 2,
       "drop_rate": 0.3,
       "vocab_size": 50257,
       "qkv_bias": False,
       "token_count": 4,
    }
    GPT_CONFIG_LARGE= {
       "emb_dim": 1280,
       "n_heads": 20,
       "n_layers": 36,
       "context_length": 1024,
       "batch_size": 2,
       "drop_rate": 0.3,
       "vocab_size": 50257,
       "qkv_bias": False,
       "token_count": 4,
    }
    GPT_CONFIG_XL = {
       "emb_dim": 1600,
       "n_heads": 25,
       "n_layers": 48,
       "context_length": 1024,
       "batch_size": 2,
       "drop_rate": 0.3,
       "vocab_size": 50257,
       "qkv_bias": False,
       "token_count": 4,
    }
    model_med = GPTModel(GPT_CONFIG_MED)
    model_large = GPTModel(GPT_CONFIG_LARGE)
    model_xl = GPTModel(GPT_CONFIG_XL)

    total_med = count_parameters(model_med)
    total_params_med = total_med - count_parameters(model_med.out_head)
    size_med_model = total_med * 4 / (1024 * 1024)

    total_large = count_parameters(model_large)
    total_params_large = total_large - count_parameters(model_large.out_head)
    size_large_model = total_large * 4 / (1024 * 1024)

    total_xl = count_parameters(model_xl)
    total_params_xl = total_xl - count_parameters(model_xl.out_head)
    size_xl_model = total_xl * 4 / (1024 * 1024)

    print("Number of trainable medium model parameters considering weight tying: "
        f"{total_params_med:,}")
    print(f"Size of the medium model: {size_med_model:.2f} MB")

    print("Number of trainable large model parameters considering weight tying: "
        f"{total_params_large:,}")
    print(f"Size of the large model: {size_large_model:.2f} MB")

    print("Number of trainable xl model parameters considering weight tying: "
        f"{total_params_xl:,}")
    print(f"Size of the xl model: {size_xl_model:.2f} MB")

