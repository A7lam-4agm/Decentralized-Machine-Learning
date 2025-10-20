# Federated Learning Project


## Project Overview
This project implements a complete Federated Learning system with:
- Federated Learning simulation with 64 clients
- Web application for model serving
- Docker containerization
- Deployment on Chameleon Cloud

## Project Structure
**assignment2**→

**fl_simulation** → #Code for training the model with federated learning

 - report → #Project report and figures

- results→ #Training logs and data files

**web_app** → #FastAPI web application

**README.md**→ # This file



---

## Step 1: Set Up Federated Learning Simulation

**Install Dependencies**
~~~
cd fl_simulation
pip install -r requirements.txt
~~~
**Run FL Training**
~~~
python main.py
~~~
**What This Does:**
- Trains ResNet-18 on CIFAR-10 with 64 clients

- Uses Dirichlet distribution for non-IID data

- Runs 10 communication rounds

- Saves model as global_model.pth

# Step 2: Prepare Files for Deployment
**Copy Files to Chameleon Cloud**
~~~
scp -r web_app/ cc@192.5.86.218:/home/cc/
~~~
# Files Transferred:
- web_app/main.py - FastAPI application

- web_app/Dockerfile - Container configuration

- web_app/requirements.txt - Python dependencies

- web_app/global_model.pth - Trained model

# Step 3: Deploy on Chameleon Cloud
**Connect to Chameleon Instance**
~~~
ssh -i ~/.ssh/private_key_chem.pem cc@192.5.86.218
~~~
**Install Docker (One-Time Setup)**
~~~
sudo apt update
sudo apt install docker.io -y
sudo usermod -aG docker cc
sudo systemctl start docker
~~~
**Build Docker Image**
~~~
cd web_app
docker build -t fl-web-app .
~~~
**Run Application**

~~~
docker run -d -p 8080:8081 --name fl-app fl-web-app
~~~
**Verify Deployment**
~~~
docker ps
curl http://localhost:8080/health
~~~
# Step 4: Access Your Application
**Web Interface**
Open in any browser:
http://192.5.86.218:8080

**API Endpoints**
- GET / - Web interface for image upload

- POST /predict - Image classification API

- GET /health - Service health check

# Step 5: Test Your Deployment
**Upload an Image**
Go to http://192.5.86.218:8080

- Click "Choose File" to upload an image

- Click "Classify Image"

- View prediction results with confidence scores

**Supported Classes:**

- airplane

- automobile

- bird

- cat

- deer

- dog

- frog

- horse

- ship

- truck

# Technical Details
**Model Information** 

* Architecture: ResNet-18 adapted for CIFAR-10

* Input Size: 32x32 RGB images

* Classes: 10 (CIFAR-10)

* Training: Federated Averaging (FedAvg)

**Deployment Specs**

* Framework: FastAPI
* Container: Docker
* Port: 8080 (external) → 8081 (internal)
* Platform: Chameleon Cloud

# Troubleshooting
**Check Container Status**

~~~
docker ps
docker logs fl-app
~~~
**Restart Application**
~~~
docker stop fl-app
docker rm fl-app
docker run -d -p 8080:8081 --name fl-app fl-web-app
~~~
**Verify Model Loading**
~~~
curl http://localhost:8080/health
~~~
# Project Deliverables Completed
✅ FL simulation with 64 clients

✅ Non-IID data distribution

✅ ResNet-18 model training

✅ FastAPI web application

✅ Docker containerization

✅ Chameleon Cloud deployment

✅ Remote accessibility

✅ Working image classification

~~~
Live Application: http://192.5.86.218:8080
~~~

- **Name**: Ahlam Abu Mismar
- **Course**: CS 595-003
- **Python Version**: 3.9
- **Chameleon IP**: 192.5.86.218
