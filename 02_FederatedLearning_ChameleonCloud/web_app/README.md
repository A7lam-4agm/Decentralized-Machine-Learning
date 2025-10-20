# 🌐 Federated Learning Web App

This FastAPI-based web app serves the **global model** trained in the Federated Learning project.  
It allows users to upload images and get real-time classification results from the model.

---

## ⚙️ Quick Start

### Install Dependencies
~~~
pip install -r requirements.txt

~~~


## Run Locally
~~~
uvicorn main:app --host 0.0.0.0 --port 8080
Then open 👉 http://127.0.0.1:8080
~~~

## 🐳 Docker Deployment
~~~
docker build -t fl-web-app .
docker run -d -p 8080:8081 --name fl-app fl-web-app
~~~

## 🌍 Live App
~~~
Deployed on Chameleon Cloud:
👉 http://192.5.86.218:8080
~~~
