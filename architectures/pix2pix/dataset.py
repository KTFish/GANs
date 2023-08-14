import config
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MapDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        super().__init__()
        self.root_dir = root_dir

        # List all files from root directory
        self.list_files = os.listdir(self.root_dir)

    def __len__(self) -> int:
        return len(self.list_files)

    def __getitem__(self, index):
        # Get the image
        image_name = self.list_files[index]
        image_path = os.path.join(self.root_dir, image_name)
        image = np.array(Image.open(image_path))

        # Split the double image into separate images
        input_image, target_image = image[:, :600, :], image[:, 600:, :]

        # Augmentations
        augmentations = config.both_transform(image - input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image
