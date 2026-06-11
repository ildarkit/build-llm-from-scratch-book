import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out,
      context_length, dropout, num_heads, qkv_bias=False):
    super().__init__()
    assert (d_out % num_heads == 0), \
      "d_out must be divisible by num_heads"
    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer(
        "mask",
        torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

  def forward(self, x):
    b, num_tokens, d_in = x.shape
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)
    queries = queries.view(
      b, num_tokens, self.num_heads, self.head_dim
    )
    keys = keys.transpose(1, 2)
    queries = queries.transpose(1, 2)
    values = values.transpose(1, 2)
    attn_scores = queries @ keys.transpose(2, 3)
    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
    attn_scores.masked_fill_(mask_bool, -torch.inf)
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)
    context_vec = (attn_weights @ values).transpose(1, 2)
    context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
    context_vec = self.out_proj(context_vec)
    return context_vec


def test_gpt2_attention():
    print("🤖 Запуск тестирования модуля MultiHeadAttention...")
    
    d_in = 768            # Размер входных эмбеддингов
    d_out = 768           # Размер выходных эмбеддингов
    num_heads = 12         # 12 целей (голов) внимания
    context_length = 1024  # Максимальная длина контекста GPT-2
    batch_size = 2 
    dropout=0.3
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Вычисления будут производиться на: {device.upper()}")
    
    try:
        mha = MultiHeadAttention(
            d_in=d_in, 
            d_out=d_out, 
            context_length=context_length, 
            dropout=dropout,
            num_heads=num_heads
        ).to(device)
        print("✅ Модуль успешно инициализирован.")
    except Exception as e:
        print(f"❌ Ошибка при создании экземпляра класса: {e}")
        return

    batch = torch.randn(batch_size, context_length, d_in, device=device)
    print(f"📦 Создан тестовый батч. Форма: {list(batch.shape)}")
    
    try:
        mha.eval() # Переводим в режим оценки (выключает dropout)
        with torch.no_grad(): # Отключаем расчет градиентов для экономии памяти
            output = mha(batch)
        print("✅ Батч успешно прошел через прямой проход (forward).")
    except Exception as e:
        print(f"❌ Ошибка во время выполнения прямого прохода: {e}")
        return

    print("\n🔍 Проверка выходных размерностей...")
    try:
        assert output.shape == (batch_size, context_length, d_out), \
            f"Неверная выходная форма! Ожидалось {(batch_size, context_length, d_out)}, \
            получили {list(output.shape)}"
        
        assert not torch.isnan(output).any(), \
                "Критическая ошибка: в выходном тензоре обнаружены NaN (ошибка деления)!"
        assert not torch.isinf(output).any(), \
                "Критическая ошибка: в выходном тензоре обнаружены бесконечности (Inf)!"
        
        print("🔥 ТЕСТ УСПЕШНО ПРОЙДЕН! Ваш модуль полностью готов к работе в составе GPT-2.")
        
    except AssertionError as error:
        print(f"❌ Тест провален: {error}")

if __name__ == "__main__":
    test_gpt2_attention()

