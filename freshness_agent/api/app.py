# app.py
from flask import Flask, request, jsonify
from PIL import Image
import pickle
import io

app = Flask(__name__)

print("Loading expanded_model_v4...")
with open("models/expanded_model_v4.pkl", "rb") as f:
    pipeline = pickle.load(f)
print(f"✅ Model ready. Classes: {len(pipeline.model.config.id2label)}")


def get_result(label, score):
    if "Fresh" in label:
        item_name = label.replace("Fresh", "").strip()
    else:
        item_name = label.replace("Rotten", "").strip()

    if "Fresh" in label:
        if score >= 70:
            rating, status, shelf_life = round(0.7 + (score/1000), 2), "FRESH", "5-7 days"
        else:
            rating, status, shelf_life = round(0.5 + (score/1000), 2), "MODERATE", "2-3 days"
    elif "Rotten" in label:
        if score >= 70:
            rating, status, shelf_life = round(score/1000, 2), "ROTTEN", "Not good for health"
        else:
            rating, status, shelf_life = round(0.3 + (score/1000), 2), "MODERATE", "2-3 days"
    else:
        rating, status, shelf_life = 0.0, "UNKNOWN", "N/A"

    return item_name, status, rating, shelf_life


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Image not found!"}), 400

    file  = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    result = pipeline(image)
    label  = result[0]['label']
    score  = round(result[0]['score'] * 100, 2)

    item_name, status, rating, shelf_life = get_result(label, score)

    return jsonify({
        "item"      : item_name,
        "status"    : status,
        "rating"    : f"{rating} / 1.0",
        "shelf_life": shelf_life,
        "confidence": score
    })


@app.route('/classes', methods=['GET'])
def classes():
    return jsonify({
        "total"  : len(pipeline.model.config.id2label),
        "classes": list(pipeline.model.config.id2label.values())
    })


if __name__ == '__main__':
    app.run(debug=True)