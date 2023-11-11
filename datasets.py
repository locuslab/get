import os
import torch


def find_files(base_dir, world_size=1, rank=0):
    path_list = os.listdir(base_dir)

    sort_key = lambda f_name: int(f_name.split('.')[0])
    path_list.sort(key=sort_key)
        
    assert len(path_list) % world_size == 0

    for i, f in enumerate(path_list):
        if i % world_size == rank:
            f_path = os.path.join(base_dir, f)
            yield f_path


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, 
            data_dir,
            world_size=1,
            rank=0,
            ):
        super().__init__()
        
        self.world_size = world_size
        self.data_dir = data_dir

        latent_dir = os.path.join(data_dir, 'latent')
        image_dir = os.path.join(data_dir, 'image')
        
        Z = list()
        for f in find_files(latent_dir, world_size, rank):
            z = torch.load(f)
            Z.append(z)
        Z = torch.cat(Z, dim=0)
        
        X = list()
        for f in find_files(image_dir, world_size, rank):
            x = torch.load(f)
            X.append(x)
        X = torch.cat(X, dim=0)
 
        assert len(Z) == len(X)
        
        self.Z = Z
        self.X = X

    def __len__(self):
        return len(self.X) * self.world_size

    def __getitem__(self, idx):
        idx = idx // self.world_size
        z, x = self.Z[idx], self.X[idx]
        
        return z, x


class PairedCondDataset(torch.utils.data.Dataset):
    def __init__(self, 
            data_dir,
            world_size=1,
            rank=0,
            ):
        super().__init__()
        
        self.world_size = world_size
        self.data_dir = data_dir

        latent_dir = os.path.join(data_dir, 'latent')
        image_dir = os.path.join(data_dir, 'image')
        
        Z = list()
        C = list()
        for f in find_files(latent_dir, world_size, rank):
            z, c = torch.load(f)
            Z.append(z)
            C.append(c)
        Z = torch.cat(Z, dim=0)
        C = torch.cat(C, dim=0)

        X = list()
        for f in find_files(image_dir, world_size, rank):
            x = torch.load(f)
            X.append(x)
        X = torch.cat(X, dim=0)
 
        assert len(Z) == len(X)
        
        self.Z = Z
        self.C = C
        self.X = X

    def __len__(self):
        return len(self.X) * self.world_size

    def __getitem__(self, idx):
        idx = idx // self.world_size
        z, c, x = self.Z[idx], self.C[idx], self.X[idx]
        
        return z, x, c


class EpochPairedDataset(torch.utils.data.Dataset):
    def __init__(self, 
            data_dir, 
            total=10,
            epoch=0,
            world_size=1,
            rank=0,
            ):
        super().__init__()
        
        self.world_size = world_size

        data_dir = data_dir + f'-{epoch%total}'
        self.data_dir = data_dir

        latent_dir = os.path.join(data_dir, 'latent')
        image_dir = os.path.join(data_dir, 'image')
        
        Z = list()
        for f in find_files(latent_dir, world_size, rank):
            z = torch.load(f)
            Z.append(z)
        Z = torch.cat(Z, dim=0)
        
        X = list()
        for f in find_files(image_dir, world_size, rank):
            x = torch.load(f)
            X.append(x)
        X = torch.cat(X, dim=0)
 
        assert len(Z) == len(X)

        self.Z = Z
        self.X = X

    def __len__(self):
        return len(self.X) * self.world_size

    def __getitem__(self, idx):
        idx = idx // self.world_size
        z, x = self.Z[idx], self.X[idx]
        
        return z, x
 
