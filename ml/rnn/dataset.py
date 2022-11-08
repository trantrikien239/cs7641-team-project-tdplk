from torch.utils.data import Dataset, DataLoader
import torch

class EssayDataset(Dataset):
    def __init__(self, data, input_col, task_cols):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.input_col = input_col
        self.task_cols = task_cols

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # return x, y, s - s is the length of the input sequence
        
        x = self.data.loc[idx,self.input_col]
        s = len(x)
        y = self.data.loc[idx,self.task_cols].tolist()
        
        return x, y, s

# Prepare dataloaders
def padding_tensor(arr, maxlen, dtype):
    padded_sess = torch.ones(len(arr), maxlen, dtype=dtype) * 400001
    
    for i in range(len(arr)):
        padded_sess[i, :len(arr[i])] = torch.tensor(arr[i])
    
    return padded_sess 

def essay_collate_fn(batch):
    inputs = [x[0] for x in batch]
    entity_seq_len = [x[2] for x in batch]
    
    labels = [x[1] for x in batch]

    maxlen = max(entity_seq_len)
        
    padded_inputs = padding_tensor(inputs, maxlen, dtype=torch.int32)

    return (padded_inputs, torch.tensor(entity_seq_len)), torch.tensor(labels)

