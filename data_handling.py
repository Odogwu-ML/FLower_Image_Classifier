from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def data_handler(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Load the datasets with ImageFolder and apply the transforms to it
    train_image_datasets = datasets.ImageFolder(f'{train_dir}', transform=data_transforms["train"])
    valid_image_datasets = datasets.ImageFolder(f'{valid_dir}', transform=data_transforms["valid"])
    test_image_datasets = datasets.ImageFolder(f'{test_dir}', transform=data_transforms["test"])


    # Using the image datasets and the transforms, define the dataloaders
    train_data_loader = DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    val_data_loader = DataLoader(valid_image_datasets, batch_size=32)
    test_data_loader = DataLoader(test_image_datasets, batch_size=32)
    
    return train_data_loader, val_data_loader, test_data_loader, train_image_datasets

