import openai
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Set OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Function to generate a caption for the meme
def generate_caption(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can also use 'gpt-3.5-turbo' or other models
        prompt=prompt,
        max_tokens=50,
        temperature=0.7
    )
    caption = response.choices[0].text.strip()
    return caption

# Function to load an image and perform object classification (using MobileNetV2 model)
def classify_image(image_url):
    print(f"Fetching image from URL: {image_url}")
    response = requests.get(image_url)
    print(f"Response status code: {response.status_code}")
    if response.status_code != 200:
        raise ValueError("Failed to fetch the image. Please check the URL.")
    
    # Print the first 100 characters of the content to inspect it
    print(f"Response content (first 100 bytes): {response.content[:100]}")
    
    try:
        img = Image.open(BytesIO(response.content)).convert("RGB")  # Ensure valid image format
    except UnidentifiedImageError:
        raise ValueError("The URL does not point to a valid image file.")
    
    # Prepare the image for classification (resize, normalize, etc.)
    img_resized = img.resize((224, 224))  # MobileNetV2 expects 224x224 images
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for MobileNetV2
    
    # Classify the image using MobileNetV2
    model = MobileNetV2(weights="imagenet")
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    
    # Return the top predicted class
    top_class = decoded_predictions[0][1]  # Class label (e.g., 'zebra', 'cat', etc.)
    return top_class, img

# Function to create meme
def create_meme(image_url):
    # Get a funny caption using GPT-3 based on image context
    top_class, image_classifications = classify_image(image_url)
    
    # Use the top classification result to generate a context-based prompt
    prompt = f"Write a humorous meme caption for an image of a {top_class}. The meme should be funny and related to popular culture."
    
    # Generate caption
    caption = generate_caption(prompt)
    
    # Fetch the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    
    # Add the caption to the image (this part can be customized)
    img = img.convert("RGBA")
    width, height = img.size
    text_color = (255, 255, 255)  # White text
    font = ImageFont.load_default()  # Use default font
    
    # Draw text on image
    draw = ImageDraw.Draw(img)
    text_width, text_height = draw.textsize(caption, font=font)
    text_position = ((width - text_width) // 2, height - text_height - 10)
    draw.text(text_position, caption, fill=text_color, font=font)
    
    # Save or display the meme
    img.show()  # This will display the meme
    img.save("generated_meme.png")  # Save the meme to a file

# Example usage
image_url = 'https://example.com/path_to_image.jpg'  # Replace with an actual image URL
create_meme(image_url)

