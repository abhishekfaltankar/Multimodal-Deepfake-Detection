import os

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms

class DeepfakeDataset(Dataset):
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):

        image_path = self.image_paths[index]

        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return image, label

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        self.image_paths = []

        self.labels = []
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        classes = {
            "real": 0,
            "fake": 1
        }

        for class_name, label in classes.items():

            class_path = os.path.join(dataset_path, class_name)

            if not os.path.exists(class_path):
                continue

            for video_folder in os.listdir(class_path):

                video_path = os.path.join(class_path, video_folder)

                if not os.path.isdir(video_path):
                    continue

                for image_name in os.listdir(video_path):

                    image_path = os.path.join(video_path, image_name)

                    self.image_paths.append(image_path)
                    self.labels.append(label)
                    
if __name__ == "__main__":

    dataset = DeepfakeDataset("dataset/processed/faces")

    print("Total Images:", len(dataset))

    image, label = dataset[0]

    print("Image Shape:", image.shape)

    print("Label:", label)