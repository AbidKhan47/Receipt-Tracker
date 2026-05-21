import os
import io
import base64
import re
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

app = Flask(__name__)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/annotate_receipt', methods=['POST'])
def annotate_receipt():
    if 'photo' not in request.files or 'tax_rate' not in request.form:
        return jsonify({"error": "Missing photo or tax_rate"}), 400
    photo = request.files['photo']
    try:
        tax_rate = float(request.form['tax_rate'])
    except ValueError:
        return jsonify({"error": "Invalid tax rate"}), 400
    try:
        img = Image.open(photo.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image file"}), 400

    if not os.environ.get("GEMINI_API_KEY"):
        return jsonify({"error": "GEMINI_API_KEY is not configured on the server."}), 500

    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = (
        "Analyze this receipt and extract the amount before tax. "
        "Return only one numeric value and nothing else. "
        "If there are multiple numbers, choose the subtotal or amount before tax."
    )

    def parse_amount_from_text(text):
        if not text:
            raise ValueError("Empty model response")
        cleaned_text = text.replace('$', '').replace(',', '')
        match = re.search(r'(?<!\\w)(\\d+(?:\\.\\d+)?)', cleaned_text)
        if not match:
            raise ValueError(f"No numeric value found in model response: {text}")
        return float(match.group(1))

    try:
        response = model.generate_content([prompt, img])
        amount_before_tax = parse_amount_from_text(response.text)
    except Exception as e:
        print(f"Error extracting pre-tax amount: {str(e)}")
        return jsonify({"error": f"Failed to extract the pre-tax amount. {str(e)}"}), 500
    new_total_amount = amount_before_tax * (1 + (tax_rate / 100))
    draw = ImageDraw.Draw(img)
    annotation_text = f"If tax were {tax_rate}%, the total amount would be ${new_total_amount:.2f}."
    try:
        font = ImageFont.truetype("arial.ttf", size=max(24, img.width // 30))
    except IOError:
        font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), annotation_text, font=font)
    text_width = right - left
    text_height = bottom - top
    img_width, img_height = img.size
    x = (img_width - text_width) // 2
    y = img_height - text_height - (img_height // 10) 
    padding = 15
    draw.rectangle([x - padding, y - padding, x + text_width + padding, y + text_height + padding], fill="yellow")
    draw.text((x, y), annotation_text, fill="red", font=font)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return jsonify({"amount_before_tax": amount_before_tax, "receipt": {"data": img_base64, "mimeType": "image/jpeg"}})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
