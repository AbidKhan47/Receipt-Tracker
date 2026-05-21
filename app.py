import os
from dotenv import load_dotenv
import io
import base64
import re
import json
import traceback
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import google.generativeai as genai

app = Flask(__name__)
# Load environment variables from .env when present (local development)
load_dotenv()

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

    def prepare_receipt_image(source_image):
        prepared_image = source_image.convert("RGB")
        prepared_image = ImageOps.exif_transpose(prepared_image)
        prepared_image = ImageOps.autocontrast(prepared_image)
        prepared_image = prepared_image.filter(ImageFilter.SHARPEN)
        if prepared_image.width < 1600:
            scale = max(1.0, 1600 / float(prepared_image.width))
            new_size = (int(prepared_image.width * scale), int(prepared_image.height * scale))
            prepared_image = prepared_image.resize(new_size, Image.Resampling.LANCZOS)
        return prepared_image

    prepared_img = prepare_receipt_image(img)

    if not os.environ.get("GEMINI_API_KEY"):
        return jsonify({"error": "GEMINI_API_KEY is not configured on the server."}), 500

    model = genai.GenerativeModel('gemini-3-flash-preview')
    prompt = (
        "Analyze this receipt and extract the amount before tax. "
        "Return only one numeric value and nothing else. "
        "If there are multiple numbers, choose the subtotal or amount before tax."
    )
    json_prompt = (
        "Analyze this receipt and return valid JSON only in this exact format: "
        '{"amount_before_tax": 118.00}. '
        "Use the subtotal or amount before tax. Do not include any extra text."
    )

    def parse_amount_from_text(text):
        if not text:
            raise ValueError("Empty model response")
        cleaned_text = text.strip().replace('$', '').replace(',', '')
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.strip("`")
            cleaned_text = cleaned_text.replace("json", "", 1).strip()
        try:
            parsed_json = json.loads(cleaned_text)
            if isinstance(parsed_json, dict) and "amount_before_tax" in parsed_json:
                return float(parsed_json["amount_before_tax"])
        except Exception:
            pass
        match = re.search(r'(?<!\w)(\d+(?:\.\d+)?)', cleaned_text)
        if not match:
            raise ValueError(f"No numeric value found in model response: {text}")
        return float(match.group(1))

    response_text = None
    retry_text = None
    try:
        response = model.generate_content([prompt, prepared_img])
        response_text = getattr(response, "text", None)
        if not response_text and getattr(response, "candidates", None):
            parts = response.candidates[0].content.parts
            response_text = " ".join(getattr(part, "text", "") for part in parts)
        try:
            amount_before_tax = parse_amount_from_text(response_text)
        except Exception:
            retry_response = model.generate_content([json_prompt, prepared_img])
            retry_text = getattr(retry_response, "text", None)
            if not retry_text and getattr(retry_response, "candidates", None):
                parts = retry_response.candidates[0].content.parts
                retry_text = " ".join(getattr(part, "text", "") for part in parts)
            amount_before_tax = parse_amount_from_text(retry_text)
    except Exception as e:
        print("Error extracting pre-tax amount:", str(e))
        traceback.print_exc()
        try:
            print("response_text:", repr(response_text))
        except NameError:
            print("response_text: <not set>")
        try:
            print("retry_text:", repr(retry_text))
        except NameError:
            print("retry_text: <not set>")
        # Include model responses in the JSON error for easier local debugging
        debug_info = {
            "response_text": response_text,
            "retry_text": retry_text,
        }
        return jsonify({"error": f"Failed to extract the pre-tax amount. {str(e)}", "debug": debug_info}), 500
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
