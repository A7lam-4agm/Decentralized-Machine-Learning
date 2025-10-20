"""
Web Application for Federated Learning Model Serving
CS 595 Assignment 2

"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import torch
import torch.nn as nn
from PIL import Image
import io
import torchvision.transforms as transforms
import torchvision.models as models
from datetime import datetime


# Configuration
NUM_CLASSES = 10
IMAGE_SIZE = 32
SERVER_PORT = 8001

# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Create FastAPI app
app = FastAPI(
    title="Federated Learning Model Server",
    description="Web app for serving trained FL model",
    version="1.0.0"
)

# Global model variable
model = None


def load_model():
    """
    Load the trained ResNet-18 model from FL training.
    Uses same architecture as FL simulation.
    """
    global model
    try:
        # Create ResNet-18 with CIFAR-10 adaptations
        model = models.resnet18(weights=None)

        # Adapt for CIFAR-10 (32x32 images)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # Remove maxpool for smaller images
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # 10 classes

        # Load trained weights from FL simulation
        model_path = 'global_model.pth'
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        print("✅ Trained ResNet-18 model loaded successfully")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None


def preprocess_image(image_bytes):
    """
    Convert uploaded image to tensor for model input.

    Args:
        image_bytes: Image file bytes

    Returns:
        Preprocessed image tensor
    """
    try:
        # Open and convert image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Resize and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor

    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")


# Load model when app starts
load_model()


@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    """
    Serve the main page with image upload form.
    """
    html_content = """
    <html>
        <head>
            <title>Federated Learning Model Server</title>
            <style>
                body { font-family: Arial; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .upload-form { 
                    border: 2px dashed #ccc; padding: 30px; 
                    text-align: center; margin: 20px 0; 
                    border-radius: 8px;
                }
                .result { 
                    background: #f5f5f5; padding: 20px; 
                    margin: 15px 0; border-radius: 5px; 
                }
                .class-list {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 5px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Federated Learning Model Server</h1>
                <p>Upload an image for CIFAR-10 classification</p>
                
                <div class="upload-form">
                    <h3>Upload Image for Classification</h3>
                    <form action="/predict" method="post" enctype="multipart/form-data">
                        <input type="file" name="image" accept="image/*" required>
                        <br><br>
                        <input type="submit" value="Classify Image">
                    </form>
                </div>
                
                <div class="result">
                    <h4>Supported Classes:</h4>
                    <div class="class-list">
    """

    # Add class names in two columns
    for i, name in enumerate(CLASS_NAMES):
        if i % 2 == 0:
            html_content += f"<div><strong>{name}</strong>"
        else:
            html_content += f"<strong>{name}</strong></div>"

    html_content += """
                    </div>
                </div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict")
async def classify_image(image: UploadFile = File(...)):
    """
    Handle image classification requests.

    Args:
        image: Uploaded image file

    Returns:
        JSON with prediction results
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(503, "Model not available")

    # Validate file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")

    try:
        # Read and preprocess image
        image_data = await image.read()
        input_tensor = preprocess_image(image_data)

        # Get model prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item() * 100

        # Prepare response
        result = {
            "predicted_class": CLASS_NAMES[predicted_class],
            "confidence": f"{confidence:.2f}%",
            "class_id": predicted_class,
            "all_probabilities": {
                name: f"{probabilities[i].item() * 100:.2f}%"
                for i, name in enumerate(CLASS_NAMES)
            }
        }

        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    status = {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "service": "Federated Learning Model Server",
        "timestamp": datetime.now().isoformat()
    }
    return JSONResponse(content=status)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)