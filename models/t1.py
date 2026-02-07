import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional information to character embeddings.
        x shape: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)] # type: ignore
        return x

class Model(nn.Module):
    """
    t1 model ~ 25,000 parameters
    A Transformer architecture for name gender classification.
    Uses Multi-Head Attention to capture complex phonemic patterns.
    """
    name: str = "t1"

    def __init__(self) -> None:
        super().__init__()
        d_model = 32
        nhead = 4
        num_layers = 2
        dim_feedforward = 128
        
        # 27 tokens: 0 for padding, 1-26 for a-z
        self.embedding = nn.Embedding(27, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=10)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Final linear layer to classify into 2 categories: [Male, Female]
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is a list of letter indexes from start to end.
        size (batch_size, 10)
        """
        # Embed and add positional encoding
        x = self.embedding(x) # (batch_size, 10, d_model)
        x = self.pos_encoder(x) # (batch_size, 10, d_model)
        
        # Transformer pass
        output = self.transformer_encoder(x) # (batch_size, 10, d_model)
        
        # Global Average Pooling across the sequence length (dim 1)
        # This aggregates information from all characters
        pooled = output.mean(dim=1) # (batch_size, d_model)
        
        # Logits output
        final = self.fc(pooled)
        return final
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # x is a list of letter indexes from start to end
        if x.dim() == 1:
            x = x.unsqueeze(0)

        out = self.forward(x)
        return torch.softmax(out, dim=-1)
    
    def predict_name(self, name: str) -> tuple[float, float]:
        inp: torch.Tensor = to_tensor(name)
        out: torch.Tensor = self.predict(inp)
        out = out.squeeze(dim=0)
        return (out[0].item(), out[1].item())

def to_tensor(name: str) -> torch.Tensor:
    """returns a size (10) tensor containing indexes of first 10 characters"""
    name = name.lower()
    lst: list[int] = []
    for c in name:
        n: int = ord(c) - ord('a') + 1
        if n >= 1 and n <= 26:
            lst.append(n)
        else:
            lst.append(0)
    return pad(torch.tensor(lst))

def pad(name: torch.Tensor, max_len = 10) -> torch.Tensor:
    """Ensures input tensor is exactly max_len long"""
    if len(name) < max_len:
        padding = torch.zeros(max_len - len(name), dtype=torch.int)
        name = torch.cat([name, padding])
    return name[:max_len]

if __name__ == "__main__":
    m = Model()
    print(f"Model: {m.name}")
    name_to_test = "justin"
    m_p, f_p = m.predict_name(name_to_test)
    print(f"Prediction for '{name_to_test}': Male={m_p:.4f}, Female={f_p:.4f}")
