# FarmVision API 🐄 - Image-Based Cattle Recognition Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-009688)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00)

**FarmVision API** is an AI-powered system designed to automate Animal Type Classification (ATC) in dairy farming. By leveraging deep learning and computer vision, it analyzes images of cattle and buffaloes to predict essential physical traits consistently and efficiently, reducing human error and bias.

This project was built to address the inconsistencies in manual animal evaluation and is aligned with the goals of initiatives like the Rashtriya Gokul Mission.

---

## 🎯 Key Objectives

- **Automate Image Analysis:** Upload an image of cattle/buffalo and let the AI process it.
- **Extract Physical Traits:** Automatically determine the breed, sex, estimated age (years), height (inches), and weight (kg).
- **Generate ATC Scores:** Compute standardized Animal Type Classification (ATC) scores dynamically based on the traits.
- **Seamless Integration:** Provide an easy-to-use API backend for integration with frontends (like React/Flutter) or external applications (e.g., Bharat Pashudhan App).

---

## 🛠️ Technology Stack

- **Backend API Framekwork:** Python 3, FastAPI, Uvicorn
- **Machine Learning / AI:** TensorFlow, Keras, MobileNetV2 (Transfer Learning)
- **Data Processing:** Pandas, NumPy
- **Image Processing:** OpenCV, Pillow
- **Deployment:** Google Colab (originally for model training) -> Local/Cloud server hosting

---

## 🚀 Project Architecture & ML Pipeline

The Machine Learning pipeline consists of three staged hierarchical classification models:

1. **Stage 1 (Detector):** Determines whether the uploaded image contains cattle, buffalo, or is a non-cattle image.
2. **Stage 2 (Cattle Classifier):** If cattle is detected, it utilizes a fine-tuned MobileNetV2 model to classify the specific *Cattle Breed*.
3. **Stage 3 (Buffalo Classifier):** If a buffalo is detected, it routes to a specific *Buffalo Breed* classifier model.

*Note: The current API (`app.py`) focuses on Stage 1 (Cattle Detection) and Stage 2 (Breed Classification).*

---

## 📂 Project Structure

```text
Image_Based_Cattle_Recognition_Model/
│
├── app.py                # Main FastAPI application serving the predictions
├── dataset.csv           # Master dataset containing breed mapping metadata & statistics
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation (You are here)
│
├── models/               # Saved Keras models (.h5 / .keras) and class JSON mappings
│   ├── best_cattle_detector.keras
│   ├── breed_classifier.h5
│   ├── cattle_class_indices.json
│   └── ... (Other detector and breed models)
│
└── scripts/              # Training, testing, and pipeline inference scripts
    ├── pipeline.py       # Full end-to-end inference script
    ├── train_stage1.py   # Code to train the base Cattle Detector
    ├── train_stage2.py   # Code to train the Cattle Breed Classifier
    ├── train_stage3.py   # Code to train the Buffalo Breed Classifier
    └── ... (Data split and evaluation scripts)
```

---

## 💻 Installation & Setup

Follow these steps to run the FarmVision API locally.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Image_Based_Cattle_Recognition_Model.git
cd Image_Based_Cattle_Recognition_Model
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate       # On Linux/macOS
# or
venv\Scripts\activate          # On Windows
```

### 3. Install Dependencies

Ensure your `requirements.txt` contains `fastapi`, `uvicorn`, `tensorflow`, `pandas`, `numpy`, `pillow`, `python-multipart`.
```bash
pip install fastapi uvicorn tensorflow pandas numpy pillow python-multipart scikit-learn
```

### 4. Run the API Server

Start the FastAPI application using Uvicorn.

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
The server will start at `http://localhost:8000`.

---

## 📡 API Endpoints

### 1. `GET /`
**Description:** Health check endpoint to verify that the API is running.

**Response:**
```json
{
  "message": "FarmVision API is running. Use /predict/ with POST to classify images."
}
```

### 2. `POST /predict/`
**Description:** Upload an image file for prediction. It runs the dual-stage AI model and computes the ATC score.

**Features Extracted:**
- Validates if the image is cattle.
- Predicts breed.
- Estimates average traits (sex, age, height, weight) based on the `dataset.csv` lookup.
- Calculates an ATC Score.

**Request:** `multipart/form-data` containing an image file (Field name: `file`).

**Success Response (Example):**
```json
{
  "is_cattle": true,
  "cattle_confidence": 0.985,
  "breed": "Sahiwal",
  "breed_confidence": 0.942,
  "sex": "Female",
  "age_in_year": 5.2,
  "height_in_inch": 52.0,
  "weight_in_kg": 450.0,
  "ATC_score": 6.8
}
```

**Non-Cattle / Rejection Response (Example):**
```json
{
  "is_cattle": false,
  "confidence": 0.99
}
```

---

## 📚 Machine Learning Details

- **Model Architecture:** The system heavily utilizes **MobileNetV2** for spatial feature extraction, benefiting from its lightweight nature and high accuracy suitable for edge/cloud deployments. 
- **Data Augmentation:** Techniques like rotation, shearing, zooming, brightness adjustments, and horizontal flips were heavily used to make the model robust against diverse field conditions.
- **ATC Scoring Formula:** The system uses a simple weighted formula based on standard metrics: `(0.2 * age + 0.3 * height + 0.5 * weight) / 10`.

---

## 🔮 Future Enhancements

- **Buffalo Support Integration:** Expose Stage 3 (Buffalo Breed Classification) in the main REST API.
- **Advanced ATC Model:** Replace the static metadata lookups for height/weight with dedicated regression models that estimate specific metrics directly from the images.
- **Frontend App:** Fully structured React / Flutter companion app.
- **Database:** Migrate `dataset.csv` lookups to PostgreSQL or MongoDB for dynamic trait and individual cattle tracking.
