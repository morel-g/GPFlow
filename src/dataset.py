import torch


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, x, y=None, transform=None):
        "Initialization"
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        len_x = len(self.x) if self.x is not None else 0
        len_y = len(self.y) if self.y is not None else 0
        return max(len_x, len_y)

    def __getitem__(self, index):
        "Generates one sample of data"
        x_idx = self.x[index] if self.x is not None else None

        if self.y is not None:
            if self.transform:
                return self.transform(x_idx), self.transform(self.y[index])
            return x_idx, self.y[index]
        else:
            if self.transform:
                return self.transform(x_idx)
            return x_idx


class TupleDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):

        if self.dataset2 != None:
            if self.transform:
                return (
                    self.transform(self.dataset1[index]),
                    self.transform(self.dataset2[index]),
                )
            return self.dataset1[index], self.dataset2[index]
        else:

            return self.dataset1[index], None

    def __len__(self):
        return len(self.dataset1)
