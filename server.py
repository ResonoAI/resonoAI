from flask import Flask, request, jsonify
import cv2
import numpy as np
import google.generativeai as genai
import PIL.Image

app = Flask(__name__)

def cv2_to_pil(cv_image):
    return PIL.Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

@app.route('/capture', methods=['POST'])
def capture_and_process_image():
    image_data = request.json.get('image')
    
    # Convert base64 image data to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process the image using your AI model
    pil_img = cv2_to_pil(img_np)
    
    # Replace this part with your AI model code
    # For demonstration, let's just return the processed image as a string
    genai.configure(api_key="YOUR_API_KEY_HERE")
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(["What clues can you get from this Image that will help a 911 first responder?", pil_img], stream=True)
    response.resolve()
    
    return jsonify({'message': response.text})

if __name__ == '__main__':
    app.run(debug=True)
