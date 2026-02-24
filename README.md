# 🌿 AGRITECH – AI-Powered Plant Disease Detection System

A smart web application built with **Flask** that detects plant diseases from uploaded leaf images using deep learning, and provides insights like causes, cures, and prevention using **Gemini AI**. It also features user authentication, detection history tracking, and a feedback system.

---

## ✨ Features

- 🔐 **User Authentication** – Signup, login, logout with session management
- 🧠 **Disease Detection** – Uses `google/efficientnet-b0` via Hugging Face for image classification
- 🤖 **Gemini AI Integration** – Generates disease-related information like causes and cures
- 🗃 **History Tracking** – Stores prediction history per user in an SQLite database
- 💬 **Feedback System** – Accepts feedback with rating (1–5) from users and guests
- 📊 **Analytics API** – JSON endpoints for disease trends and feedback ratings

---

## 🛠 Tech Stack

| Technology            | Purpose                                               |
|-----------------------|-------------------------------------------------------|
| 🐍 Python (Flask)     | Backend, routing, and session handling                |
| 🧠 TensorFlow / Keras | Custom CNN model (`plant_disease_model_v5.h5`)        |
| 🤖 Google Gemini API  | AI-generated disease information and recommendations  |
| 💾 SQLite             | Database to store users, prediction history, feedback |
| 🖼️ PIL (Pillow)       | Image preprocessing for model input                   |
| 🌐 HTML + Jinja2      | Frontend templates and dynamic rendering              |
| 🔐 Flask Sessions     | User authentication and session management            |

---
```
## 📁 Folder Structure

Agri_Tech/
├── app.py                  # Main Flask app
├── users.db                # SQLite database
├── templates/              # Jinja2 HTML templates
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── dashboard.html
│   ├── Agri\_tech.html
│   ├── feedback.html
│   └── history.html
├── static/
│   └── uploads/            # Uploaded leaf images
```


---

## ⚙️ Setup Instructions

### 1. 📥 Clone the repository
```bash
git clone https://github.com/B-koushik-09/agritech.git
cd Agri_Tech
````

### 2. 🧪 Set up a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate         # On Windows
# OR
source venv/bin/activate      # On macOS/Linux

pip install -r requirements.txt
```

### 3. 🔑 Configure Gemini API Key

Open `app.py` and set your Gemini API key:

```python
GEMINI_API_KEY = "your-google-api-key"
```

### 4. ▶️ Run the Application

```bash
python app.py
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🔗 Key Routes

| Route                   | Function                                |
| ----------------------- | --------------------------------------- |
| `/`                     | Homepage                                |
| `/login`, `/signup`     | User authentication                     |
| `/detect`               | Disease detection form                  |
| `/predict`              | Upload image and get prediction         |
| `/history`              | View past predictions                   |
| `/feedback`             | Submit feedback                         |
| `/api/history`          | JSON: Disease prediction history counts |
| `/api/feedback-ratings` | JSON: Feedback rating stats             |

---

## 📝 Notes

* Ensure internet access is enabled (for Gemini AI responses)
* Works best with clear, high-resolution leaf images
* You can upgrade from SQLite to MySQL/PostgreSQL for production
* Feedback from guests is stored with email: `guest` and name: `Anonymous`

---

## 🤝 Contributing

Pull requests are welcome. Feel free to fork and improve the project.

---



> 👨‍💻 Developed by [B-koushik-09](https://github.com/B-koushik-09/agritech) 

````
