import argparse
from process_image import process_img
from imshow import imshow
from torchvision import  models
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import OrderedDict
from process_image import process_img


parser = argparse.ArgumentParser()
parser.add_argument("image_pth", help="path for image")
parser.add_argument("model_checkpoint", help="loaded model checkpoint")
parser.add_argument("--top_k", help="top clases", type=int)
parser.add_argument("--category_names", help="mapping of classes to indices")
parser.add_argument("--gpu")

args = parser.parse_args()

device = torch.device("cuda" if args.gpu == "yes" else "cpu")


# Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(path):
    
    if torch.cuda.is_available() and args.gpu =="yes":
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
        
    checkpoint = torch.load(path, map_location=map_location)
  
    # Rebuild model
    new_model = models.vgg19(pretrained=True)
    for param in new_model.parameters():
        param.requires_grad = False
    
    new_model.class_to_idx = checkpoint["class_to_idx"]
    
    classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(checkpoint["inputs"], checkpoint["fc1_output"])),
        ("ReLU", nn.ReLU()),
        ("fd1", nn.Dropout(p=0.5)),
        ("fc2", nn.Linear(checkpoint["fc1_output"], checkpoint["fc2_output"])),
        ("ReLU", nn.ReLU()),
        ("fd2", nn.Dropout(p=0.5)),
        ("fc3", nn.Linear(checkpoint["fc2_output"], checkpoint["outputs"])),
        ("Output", nn.LogSoftmax(dim=1))
    ]))
    
    # update classifier for pretrained network
    new_model.classifier = classifier
    
    new_model.load_state_dict(checkpoint["state_dict"])
    
    new_model = new_model.to(device) # move model to gpu
    
    return new_model


with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)


def predict(image_path, model_path, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
  
    # Process image
    img = process_img(image_path)
    model = load_checkpoint(model_path)
    model.eval
    
    model = model.to(device) # moves model to gpu
    # convert image to pytorch tensor
    tensor_img = torch.from_numpy(img).type(torch.FloatTensor)
    
    # Add batch of size 1 to image
    m_input = tensor_img.unsqueeze(0)
    m_input = m_input.to(device)
    # Probs
    output = model(m_input)
    ps = torch.exp(output)
    
    # Top probs
    top_p, top_c = ps.topk(topk)
    top_p = top_p.detach().cpu().numpy().tolist()[0] 
    top_c = top_c.detach().cpu().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[cl] for cl in top_c]
    top_flowers = [cat_to_name[idx_to_class[cl]] for cl in top_c]
    return top_p, top_labels, top_flowers


# Make prediction

image_path = args.image_pth
model_path = args.model_checkpoint
topk = args.top_k

probs, class_, flowers = predict(image_path, model_path, topk)

# Print predictions
print("**************** Image Predicion ************")
print("Class              Probability")
print("-----------------------------------")

j = 0
for i in class_:
    print(i + f"                 {probs[j] * 100}")
    j+=1

