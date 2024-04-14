import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, mode: str = 'train'):
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        # Load data according to the mode
        if self.mode == 'train':
            self.data = torch.load(data_dir + 'train_data.pt')
            self.labels = torch.load(data_dir + 'train_labels.pt')
        elif self.mode == 'val':
            self.data = torch.load(data_dir + 'val_data.pt')
            self.labels = torch.load(data_dir + 'val_labels.pt')
        elif self.mode == 'test':
            self.data = torch.load(data_dir + 'test_data.pt')
            self.labels = torch.load(data_dir + 'test_labels.pt')
        else:
            raise ValueError("Invalid mode. Mode should be one of 'train', 'val', or 'test'.")

    def __getitem__(self, index):
        # Get data and label at the specified index
        data = self.data[index]
        label = self.labels[index]

        # Apply transform to your data if provided
        if self.transform is not None:
            data = data.float()
            data = self.transform(data)

        # Normalize input data to [0,1]
        data = data.float() / 255.0

        return data, label

    def __len__(self):
        return len(self.data)