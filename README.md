# 🍔 Food Vision — Image Classification API

## 📌 Overview

Food Vision is a deep learning-based image classification system that predicts food categories from images using a trained CNN model.

The application is built using FastAPI and containerized with Docker for easy deployment.

---

## 🚀 Features

* Image classification using deep learning
* FastAPI backend
* Dockerized environment
* Swagger UI for testing
* Real-time predictions

---

## 🧠 Tech Stack

* Python
* FastAPI
* Uvicorn
* PyTorch
* Docker

---

## ⚙️ Setup

### Clone repository

```bash
git clone https://github.com/vidit-16/food-vision.git
cd food-vision
```

---

### Build Docker image

```bash
docker build -t foodvision .
```

---

### Run container

```bash
docker run -p 8000:8000 foodvision
```

---

### Access API Docs

Open:

```
http://localhost:8000/docs
```

---

## 🧪 Usage

### Endpoint

`POST /predict`

### Input

* Upload image file

### Output

* Predicted food class

---

## 🧠 How it Works

* Image is uploaded via API
* Preprocessing applied (resize, normalize)
* Model performs inference
* Prediction returned as response

---

## 📦 Deployment

The project is containerized using Docker for consistent and portable execution.

---

## ⚠️ Note

Model file is large (~90MB), so GitHub may show warnings.

---

## 👨‍💻 Author

Vidit Choudhary
