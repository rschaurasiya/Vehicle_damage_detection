from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms


train_model = None
class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']


class CarClassifierwithResNet50(nn.Module):
  def __init__ (self,num_classes =6):
    super(). __init__()
    self.model = models.resnet50(weights = 'DEFAULT')

    # Freeze all layers except the final fully connected Layer
    for param in self.model.parameters():
      param.requires_grad = False

    # Unfreeze layer4 and fc layers
    for param in self.model.layer4.parameters():
      param.requires_grad = True


    self.model.classifier = nn.Sequential(
        nn.Dropout(0.399),
        nn.Linear(self.model.fc.in_features,num_classes)
    )

  def forward(self,x):
    x = self.model(x)
    return x

def predict(uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)

    global train_model

    if  train_model is None:
        train_model = CarClassifierwithResNet50()
        train_model.load_state_dict(torch.load("../model/saved_model.pth", map_location=torch.device("cpu"))))
        train_model.eval()

    with torch.no_grad():
      output = train_model(image_tensor)
      _, predicted_class = torch.max(output, 1)

      print("OUTPUT SHAPE:", output.shape)
      print("PREDICTED INDEX:", predicted_class.item())
      print("NUMBER OF CLASSES:", len(class_names))

      return class_names[predicted_class.item()]

      print("Predicted class index:", predicted_class.item())
      print("Number of class names:", len(class_names))

      return class_names[predicted_class.item()]

