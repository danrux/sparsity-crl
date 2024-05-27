import os
import faiss
import numpy as np
import torch
from torchvision.datasets.folder import pil_loader
from typing import Optional, Callable
import spaces




class Causal3DIdent(torch.utils.data.Dataset):
    """Load Causal3DIdent dataset"""

    FACTORS = {
        0: "object_shape",
        1: "object_ypos",
        2: "object_xpos",
        3: "object_zpos",
        4: "object_alpharot",
        5: "object_betarot",
        6: "object_gammarot",
        7: "spotlight_pos",
        8: "object_color",
        9: "spotlight_color",
        10: "background_color",
    }
    CLASSES = range(7)  # number of object shapes # we only consider teapot here
    LATENT_SPACES = {}
    for i, v in FACTORS.items():
        if i == 0:
            LATENT_SPACES[i] = spaces.DiscreteSpace(n_choices=len(CLASSES))
        else:
            LATENT_SPACES[i] = spaces.NBoxSpace(n=1, min_=-1.0, max_=1.0)

    mean_per_channel = [0.4327, 0.2689, 0.2839]
    std_per_channel = [0.1201, 0.1457, 0.1082]

    POSITIONS = [1, 2, 3]
    ROTATIONS = [4, 5, 6]
    HUES = [7, 8, 9]

    DISCRETE_FACTORS = {0: "object_shape"}

    def __init__(
        self,
        data_dir: str,
        mode: str = "train",
        transform: Optional[Callable] = None,
        loader: Optional[Callable] = pil_loader,
        latent_dimensions_to_use=range(10),
        approximate_mode: Optional[bool] = True,
        mask_prob=0.5,
        sigma=0.1,
        class_idx = int(0),
    ):
        super(Causal3DIdent, self).__init__()

        self.mask_prob = mask_prob

        self.mode = mode
        if self.mode == "val":
            self.mode = "test"

        self.sigma = sigma
        self.class_idx = class_idx
        self.root = os.path.join(data_dir, self.mode)

        self.classes = self.CLASSES
        self.latent_classes = []
        for i in self.classes:
            self.latent_classes.append(
                np.load(os.path.join(self.root, "raw_latents_{}.npy".format(i)))
            )
        self.unfiltered_latent_classes = self.latent_classes

        if latent_dimensions_to_use is not None:
            # print('not none')
            for i in self.classes:
                self.latent_classes[i] = np.ascontiguousarray(
                    self.latent_classes[i][:, latent_dimensions_to_use]
                )

        self.image_paths_classes = []
        for i in self.classes:
            max_length = int(np.ceil(np.log10(len(self.latent_classes[i]))))
            self.image_paths_classes.append(
                [
                    os.path.join(
                        self.root,
                        "images_{}".format(i),
                        f"{str(j).zfill(max_length)}.png",
                    )
                    for j in range(self.latent_classes[i].shape[0])
                ]
            )
        self.loader = loader
        self.transform = transform or (lambda x: x)

        self._index_classes = []
        for i in self.classes:
            if approximate_mode:
                _index = faiss.index_factory(
                    self.latent_classes[i].shape[1], "IVF1024_HNSW32,Flat"
                )
                _index.efSearch = 8
                _index.nprobe = 10
            else:
                _index = faiss.IndexFlatL2(self.latent_classes[i].shape[1])

            if approximate_mode:
                _index.train(self.latent_classes[i])
            _index.add(self.latent_classes[i])
            self._index_classes.append(_index)

    def __len__(self) -> int:
        return len(self.latent_classes[0]) * len(self.classes)

    def __getitem__(self, item):
        class_id = self.class_idx  # item // len(self.latent_classes[0]) # NOTE: only select teacups
        z = torch.stack(
            [
                v.normal(mean=torch.zeros(1), std=self.sigma, size=1)
                for i, v in self.LATENT_SPACES.items()
                if i != 0
            ]
        ).squeeze()

        distance_z, index_z = self._index_classes[class_id].search(
            z[None], 1
        )  
        index_z = index_z[0, 0]  

        z = self.latent_classes[class_id][index_z]
        

        return torch.from_numpy(z).float()  # , img

    def collate_fn(self, masks, mask_values, batch):
        class_id = self.class_idx
        
        masked_zs = (
            torch.einsum("ij,kj->ikj", masks, torch.stack(batch))
            + (1 - masks[:, None, :]) * mask_values
        )
        masked_zs = masked_zs.reshape(-1, masked_zs.shape[-1])

        imgs = []
        for z in masked_zs:
            distance_z, index_z = self._index_classes[class_id].search(
                z[None], 1
            )  
            index_z = index_z[0, 0]  
            z = self.latent_classes[class_id][index_z]
            path_z = self.image_paths_classes[class_id][index_z]
            img = self.transform(self.loader(path_z))
            imgs += [img]
        return masked_zs, torch.stack(imgs)
