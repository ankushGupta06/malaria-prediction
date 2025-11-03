# 🦠 Malaria Parasite Detection using Deep Learning

This project uses a **Convolutional Neural Network (CNN)** built with **TensorFlow** to detect malaria parasites in blood cell images.  
A simple **Gradio web interface** allows users to upload an image and instantly see if it’s *Parasitized* or *Uninfected*.

---

## 🚀 Features

- Detects malaria parasites in microscopic images  
- Built using TensorFlow and Keras  
- Interactive Gradio web interface  
- Lightweight and easy to deploy  

---

## 🧠 Model Overview

- **Architecture:** Convolutional Neural Network (CNN)  
- **Input Size:** 128 × 128 × 3  
- **Output:** Binary classification — *Parasitized* or *Uninfected*  
- **Framework:** TensorFlow / Keras  

---

## 📂 Project Structure
📁 malaria-parasite-detection
│
├── app.py # Gradio app for real-time prediction
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── malaria_parasite.keras # Trained model file


---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/ankushGupta06/malaria-prediction.git
cd malaria-parasite-detection

pip install -r requirements.txt

▶️ Run the App
python app.py


Then open the local Gradio link (shown in the terminal) to test the model in your browser.


🧩 Example Usage

Upload a blood smear image and get predictions:

🟥 Parasitized → Infected with malaria parasite

🟩 Uninfected → Healthy cell image

📦 Requirements

Python ≥ 3.8

TensorFlow ≥ 2.10

Gradio ≥ 4.0

🤖 Future Improvements

Add more image preprocessing

Improve model accuracy using data augmentation

🧑‍💻 Author

Ankush Gupta
📧 iamankushgupta68@gmail.com
⭐ If you like this project, consider giving it a star on GitHub!

⚠️ Disclaimer

This tool is for educational and research purposes only.
It should not be used for real medical diagnosis.
