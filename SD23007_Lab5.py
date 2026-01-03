import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
import pandas as pd

# Step 1: Create a new Streamlit application and configure page title and layout
st.set_page_config(page_title = "CPU - Based Image Classification App", page_icon = "", layout = "centered")

# Step 2: Import required libraries (done above, including Streamlit, PyTorch, Torchvision, PIL, Pandas)

# Step 3: Configure to run only on CPU settings
device = torch.device("cpu")

# Utility: Load ImageNet labels 
@st.cache_data
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    labels = response.text.strip().split("\n")
    return labels

labels = load_imagenet_labels()

# Step 4: Load pre-trained ResNet18 model and set to evaluation mode 
@st.cache_resource
def load_model():
    model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    model.to(device) # Explicitly move to CPU
    model.eval()
    return model

model = load_model()

# Step 5: Apply recommended image preprocessing  transformation 
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# Step 6: Design user interface for image upload
st.title("Simple CPU - Based Image Classification Web App")
st.write("Upload an image for classification using  pre - trained ResNet18 on CPU.")
uploaded_file = st.file_uploader("Upload an image (jpg/png)", type = ["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploade image 
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption = "Uploaded Image", use_column_width = True)

    # Step 7: Convert uploaded image to tensor and preprocess model interface without gradient computation
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dimension and move to CPU
    with torch.no_grad():
        output = model(input_batch)

    # Step 8: Apply softamax to model output and display top -5 predictions c;assses with probabilities
    probabilities = F.softmax(output[0], dim = 0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    st.subheader("Top-5 Predictions:")
    for i in range(top5_prob.size(0)):
        label = labels[top5_catid[i]]
        prob = top5_prob[i].item() * 100 # Convert to percentage
        st.write(f"**{label}**: {prob:.2f}%")

    # Step 9: Visualize intermediate prediction probabilities using a bar chart in Streamlit 
    df = pd.DataFrame({
        "Class": [labels[idx] for idx in top5_catid],
        "Probability (%)": [p.item() * 100 for p in top5_prob]
    })
    st.subheader("Prediction Probabilities Bar Chart:")
    st.bar_chart(df.set_index("Class"))
else:
    st.info("Please upload an image to classify.")