import torch
import torch.nn as nn




class m_AttentionPool2d(nn.Module):
    def __init__(self):
        super(m_AttentionPool2d, self).__init__()
        self.m_atten = nn.MultiheadAttention(embed_dim=2048, num_heads=32, bias=True, add_bias_kv=False)

    def forward(self, x):
        (op , _) = self.m_atten(query=x[:1], key=x, value=x, need_weights=False, average_attn_weights=False, is_causal=False)
        return op
    
    
    
    
model = m_AttentionPool2d()

example_inputs = torch.rand(50,20,2048)

example_outputs = model(example_inputs)
print(example_outputs.shape)  