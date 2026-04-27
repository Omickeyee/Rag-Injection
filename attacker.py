import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, model_name, pipeline_model):
        super().__init__()
        self.base = pipeline_model.model
        hidden_size = self.base.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        values = self.value_head(hidden_states).squeeze(-1)
        return logits, values