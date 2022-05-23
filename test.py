from data_loader import DATA
from model import MODEL
import torch


dat = DATA(158,300,",")

train_data = dat.load_data('/home/thales/pytorch_dkvmn/data/errex/train.csv')
teste, teste2,teste3,teste4 = train_data


model = MODEL(n_question=180,
                  batch_size=10,
                  q_embed_dim=50,
                  qa_embed_dim=100,
                  memory_size=50,
                  memory_key_state_dim=50,
                  memory_value_state_dim=100,
                  final_fc_dim=50)

model.load_state_dict(torch.load('/home/thales/pytorch_dkvmn/save (1).pt'))

print(teste2.shape)
print(teste4.shape)