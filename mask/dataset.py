import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, stateset, actionset):
        super(MyDataset,self).__init__()
        
        self.stateset = stateset
        self.actionset = actionset
    
    def __getitem__(self, index):
        states = self.stateset[index]
        actions = self.actionset[index]
        return states, actions
    
    def __len__(self):
        return len(self.stateset)

