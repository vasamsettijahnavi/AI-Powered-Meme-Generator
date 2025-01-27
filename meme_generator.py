import openai
from PIL import Image
import requests
from io import BytesIO
import random
import numpy as np
import tensorflow as tf

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

# Function to load an image and perform object classification (using a CNN model)
def classify_image(image_url):
    # Load pre-trained model (MobileNetV2 for example)
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    
    # Load and preprocess the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Decode predictions (using ImageNet classes)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Function to create meme
def create_meme(image_url):
    # Get a funny caption using GPT-3 based on image context
    image_classifications = classify_image(image_url)
    
    # Use the top classification result to generate a context-based prompt
    top_class = image_classifications[0][1]
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
