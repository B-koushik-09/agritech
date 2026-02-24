import os
import io
import json
import sqlite3
from datetime import datetime, timedelta
from collections import Counter

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS

import google.generativeai as genai

# ---------------------------
# Setup Gemini API key and model
# ---------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gen_model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gen_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            },
        )
        print("✅ Gemini model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Gemini initialization failed: {e}")
else:
    print("⚠️ Gemini API key not set or invalid. Disease info will show as 'Information not available'.")

# ---------------------------
# Flask App and Configuration
# ---------------------------
app = Flask(__name__)
app.secret_key = "secret-key"
CORS(app)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# Load Keras Model
# ---------------------------
MODEL_PATH = "plant_disease_model_v5.h5" 
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 15 

try:
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    keras_model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    keras_model.load_weights(MODEL_PATH)
    keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("✅ Model architecture rebuilt and weights loaded successfully.")

except Exception as e:
    print(f"❌ Could not load weights from '{MODEL_PATH}': {e}")
    raise e

dummy_input = np.zeros((1, 224, 224, 3))
_ = keras_model.predict(dummy_input, verbose=0)
print("✅ Model initialized and ready for prediction!")

# Load class names from JSON or fallback list
CLASS_NAMES = None
if os.path.exists("class_indices_v5.json"):
    try:
        with open("class_indices_v5.json", "r", encoding="utf-8") as f:
            class_indices = json.load(f)
            inv = {int(v): k for k, v in class_indices.items()}
            CLASS_NAMES = [inv[i] for i in range(len(inv))]
            print("✅ Loaded class names from class_indices_v5.json")
    except Exception as e:
        print("[WARN] Failed loading class_indices_v5.json:", e)

if CLASS_NAMES is None:
    CLASS_NAMES = [
        "Pepper, bell___Bacterial_spot", "Pepper, bell___healthy", "Potato___Early_blight",
        "Potato___Late_blight", "Potato___healthy", "Tomato___Bacterial_spot",
        "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites_Two-spotted_spider_mite",
        "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
    ]
    print("⚠️ CLASS_NAMES fallback used.")

# ---------------------------
# Database Initialization
# ---------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        disease TEXT,
        file_path TEXT,
        timestamp TEXT
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        name TEXT NOT NULL,
        message TEXT NOT NULL,
        rating INTEGER DEFAULT 0,
        timestamp TEXT DEFAULT (datetime('now','localtime'))
    )""")

    conn.commit()
    conn.close()

init_db()

def update_feedback_table():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute("PRAGMA table_info(feedback)")
    existing_columns = [row[1] for row in c.fetchall()]

    if "rating" not in existing_columns:
        c.execute("ALTER TABLE feedback ADD COLUMN rating INTEGER DEFAULT 0")

    if "timestamp" not in existing_columns:
        c.execute("ALTER TABLE feedback ADD COLUMN timestamp TEXT")
        c.execute("UPDATE feedback SET timestamp = datetime('now', 'localtime') WHERE timestamp IS NULL")

    conn.commit()
    conn.close()

update_feedback_table()

# ---------------------------
# Utilities
# ---------------------------
def format_disease_name(name):
    # Convert underscores and triple underscores to readable text
    return name.replace("_", " ").replace("___", " - ").title()

def get_disease_info(disease_name):
    # If Gemini model not available return static message
    if gen_model is None:
        return {
            "causes": ["Information not available"],
            "cure": ["Information not available"],
            "prevention": ["Information not available"],
            "recommendations": ["Information not available"]
        }
    try:
        # Prompt designed for structured JSON output
        prompt = (
            f"Give causes, cure, prevention and farmer recommendations for '{disease_name}' in plants.\n"
            "Return JSON with keys: causes, cure, prevention, recommendations (each a list)."
        )
        response = gen_model.generate_content(prompt)
        text = response.text.strip()
        # Try loading JSON response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback to scanning lines into lists
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return {
                "causes": lines[0:3] or ["Information not available"],
                "cure": lines[3:6] or ["Information not available"],
                "prevention": lines[6:9] or ["Information not available"],
                "recommendations": lines[9:12] or ["Information not available"]
            }
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return {
            "causes": ["Information not available"],
            "cure": ["Information not available"],
            "prevention": ["Information not available"],
            "recommendations": ["Information not available"]
        }

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/detect")
def detect():
    return render_template("Agri_tech.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        user = c.fetchone()
        conn.close()

        if user:
            session["user_email"] = user[2]
            session["user_name"] = user[1]
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
            conn.commit()
        except sqlite3.IntegrityError:
            return render_template("signup.html", error="Email already registered.")
        finally:
            conn.close()
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ---------------------------
# Predict route using Keras model
# ---------------------------

def truncate_list(lst, n=3):
    return lst[:n] if isinstance(lst, list) else lst

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        fp = request.files["file"]
        img_bytes = fp.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        pil_img_resized = pil_img.resize((224, 224))
        arr = keras_image.img_to_array(pil_img_resized)
        arr = (arr / 127.5) - 1.0  # Normalize for MobileNetV2
        arr = np.expand_dims(arr, axis=0)

        preds = keras_model.predict(arr, verbose=0)
        idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(preds[0][idx])

        if 0 <= idx < len(CLASS_NAMES):
            disease_raw = CLASS_NAMES[idx]
        else:
            disease_raw = f"class_{idx}"

        disease = format_disease_name(disease_raw)

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        pil_img.save(file_path)

        user_email = session.get("user_email", "guest")

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("INSERT INTO history (user_email, disease, file_path, timestamp) VALUES (?, ?, ?, ?)",
                  (user_email, disease, file_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        if "healthy" in disease.lower():
            info_truncated = {
                "causes": ["Your plant appears healthy."],
                "cure": ["No treatment needed."],
                "prevention": ["To keep it that way, ensure regular watering, proper sunlight, and monitor for any new spots or discoloration."],
                "recommendations": ["Continue regular care and watch for any unusual changes in leaves."]
            }
        else:
            # For diseased leaves — fetch from Gemini
            info = get_disease_info(disease)
            info_truncated = {
                "causes": truncate_list(info.get("causes", []), 3),
                "cure": truncate_list(info.get("cure", []), 3),
                "prevention": truncate_list(info.get("prevention", []), 3),
                "recommendations": truncate_list(info.get("recommendations", []), 3)
            }

        # This line should not be indented more than the rest of the function!
        return jsonify({
            "disease": disease,
            "confidence": round(confidence, 4),
            **info_truncated
        })

    except Exception as e:
        print("[PREDICT ERROR]", e)
        return jsonify({"error": str(e)}), 500

# ---------------------------
# History route with pagination and mapping
# ---------------------------
@app.route("/history")
def history():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    per_page = 15
    page = int(request.args.get("page", 1))
    offset = (page - 1) * per_page

    user_history = []
    if "user_email" in session:
        user_email = session["user_email"]
        c.execute("""SELECT id, user_email, disease, file_path, timestamp 
                     FROM history 
                     WHERE user_email=? 
                     ORDER BY timestamp DESC""", (user_email,))
        rows = c.fetchall()
        user_history = [
            {
                "id": row[0],
                "user": row[1],
                "disease": row[2],
                "image_filename": os.path.basename(row[3]),
                "timestamp": row[4]
            } for row in rows
        ]

    c.execute("SELECT COUNT(*) FROM history")
    total_records = c.fetchone()[0]
    total_pages = (total_records + per_page - 1) // per_page

    c.execute("""SELECT id, user_email, disease, file_path, timestamp 
                 FROM history 
                 ORDER BY timestamp DESC 
                 LIMIT ? OFFSET ?""", (per_page, offset))
    rows = c.fetchall()

    c.execute("SELECT email, name FROM users")
    user_map = dict(c.fetchall())

    total_history = [
        {
            "id": row[0],
            "user": user_map.get(row[1], "Guest") if row[1] != "guest" else "Guest",
            "disease": row[2],
            "image_filename": os.path.basename(row[3]),
            "timestamp": row[4]
        } for row in rows
    ]

    conn.close()

    return render_template("history.html",
                           user_history=user_history,
                           total_history=total_history,
                           current_page=page,
                           total_pages=total_pages)

# ---------------------------
# Feedback routes (unchanged)
# ---------------------------
@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        name = request.form["name"]
        message = request.form["feedword"]
        rating = int(request.form.get("rating", 0))
        email = session.get("user_email", "guest")

        conn = sqlite3.connect("users.db")
        c = conn.cursor()

        if email == "guest":
            c.execute("SELECT timestamp FROM feedback WHERE user_email='guest' ORDER BY id DESC LIMIT 1")
            last = c.fetchone()
            if last and last[0]:
                try:
                    last_time = datetime.strptime(last[0], "%Y-%m-%d %H:%M:%S")
                    if datetime.now() - last_time < timedelta(seconds=60):
                        conn.close()
                        return render_template("feedback.html", success=False, error="⚠️ Please wait before submitting again.")
                except ValueError:
                    pass

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO feedback (user_email, name, message, rating, timestamp) VALUES (?, ?, ?, ?, ?)",
                  (email, name, message, rating, timestamp))
        conn.commit()
        conn.close()
        return render_template("feedback.html", success=True)

    return render_template("feedback.html", success=False)

@app.route("/feedback-history")
def feedback_history():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    user_email = session.get("user_email")
    user_feedback = []

    if user_email:
        c.execute("SELECT name, message, rating, timestamp FROM feedback WHERE user_email=? ORDER BY id DESC", (user_email,))
        user_feedback = [{"name": row[0], "message": row[1], "rating": row[2], "timestamp": row[3]} for row in c.fetchall()]

    c.execute("SELECT name, message, rating, timestamp FROM feedback ORDER BY id DESC")
    all_feedback = [{"name": row[0], "message": row[1], "rating": row[2], "timestamp": row[3]} for row in c.fetchall()]

    c.execute("SELECT COUNT(*), AVG(rating) FROM feedback")
    stats = c.fetchone()
    total_count = stats[0]
    avg_rating = round(stats[1], 2) if stats[1] else 0

    conn.close()
    return render_template("feedback_history.html", user_feedback=user_feedback,
                           all_feedback=all_feedback, total_count=total_count, avg_rating=avg_rating)

# ---------------------------
# API Endpoints
# ---------------------------
@app.route("/api/history")
def api_history():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    if "user_email" in session:
        user_email = session["user_email"]
        c.execute("SELECT disease FROM history WHERE user_email=?", (user_email,))
        user_diseases = [row[0] for row in c.fetchall()]
        user_counter = Counter(user_diseases)
        user_data = [{"disease": name, "count": count} for name, count in user_counter.items()]
    else:
        user_data = []

    c.execute("SELECT disease FROM history")
    all_diseases = [row[0] for row in c.fetchall()]
    total_counter = Counter(all_diseases)
    total_data = [{"disease": name, "count": count} for name, count in total_counter.items()]

    conn.close()
    return jsonify({"total": total_data, "user": user_data})

@app.route("/api/feedback-ratings")
def feedback_ratings():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating")
    rows = c.fetchall()
    conn.close()

    return jsonify({str(rating): count for rating, count in rows})

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
