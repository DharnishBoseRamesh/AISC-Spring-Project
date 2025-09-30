import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # (3, 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (64, 128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # (128, 256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # (256, 512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # (512, 512)


        # Final shape after pool: (B, 256, 7, 7)
        self.flatten = nn.Flatten() #(256 x 7 x 7) -> (1 x 1 x (256 * 7 * 7))
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes) #num_classes: number of output types

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # (3 x 224 x 224) -> (B, 64, 112, 112)
        x = self.pool(self.relu(self.conv2(x)))  # (B, 64, 112, 112) -> (B, 128, 56, 56)
        x = self.pool(self.relu(self.conv3(x)))  # (B, 128, 56, 56) -> (B, 256, 28, 28)
        x = self.pool(self.relu(self.conv4(x)))  # (B, 256, 28, 28) -> (B, 512, 14, 14)
        x = self.pool(self.relu(self.conv5(x)))  # (B, 512, 14, 14) -> (B, 512, 7, 7)


        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
#load model
@st.cache_resource
def load_model():
    model = BrainTumorCNN()
    model.load_state_dict(torch.load("best_brain_tumor_model (1).pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

#define transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) #from train_transform in tumporbp.ipynb

label_map = {
    0: 'notumor',
    1: 'glioma', 
    2: 'meningioma',
    3: 'pituitary'
}

st.title("Brain Tumor Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)

    input_tensor = transform(image).unsqueeze(0) #add batch dimension
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()

    st.write(f"Predicted Class: {label_map[prediction]}")