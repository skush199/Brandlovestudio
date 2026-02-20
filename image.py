import openai
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
import os
import io
 
# Set up OpenAI API key
openai.api_key = 'your-openai-api-key'  # Replace with your actual OpenAI API key
 
# Set up Google Vision API credentials
credentials = service_account.Credentials.from_service_account_file(
    'path_to_your_google_service_account_key.json'  # Replace with the path to your Google service account JSON
)
client = vision.ImageAnnotatorClient(credentials=credentials)
 
# Function to process the image and identify objects using Google Vision API
def process_image_with_vision(image_path):
    # Load image into memory
    with open(image_path, 'rb') as img_file:
        content = img_file.read()
 
    # Create an image instance for Vision API
    image = vision.Image(content=content)
 
    # Perform label detection
    response = client.label_detection(image=image)
 
    if response.error.message:
        raise Exception(f"Error with Google Vision API: {response.error.message}")
 
    # Extract the labels from the response
    labels = [label.description for label in response.label_annotations]
    return labels
 
# Function to send the detected objects to OpenAI for generating detailed descriptions
def generate_description_with_openai(labels):
    # Create a detailed prompt for OpenAI based on the detected labels
    prompt = f"Please describe the following objects in detail as if you are explaining them in a sentence: {', '.join(labels)}."
    # Send the prompt to OpenAI API to generate a description
    response = openai.Completion.create(
        model="gpt-4.1",  # You can change to a more suitable model like GPT-4 if available
        prompt=prompt,
        max_tokens=200,  # Adjust the max tokens for a longer or shorter description
        temperature=0.7  # Control the creativity of the response
    )
    description = response['choices'][0]['text'].strip()  # Extract the response text
    return description
 
# Main function to run the image processing and description generation
def main(image_path):
    # Step 1: Process the image to detect objects using Google Vision API
    labels = process_image_with_vision(image_path)
    # Step 2: Generate a detailed description using OpenAI based on the detected labels
    description = generate_description_with_openai(labels)
    # Step 3: Print out the detected labels and the generated description
    print("Detected Objects:", labels)
    print("Generated Description:", description)
 
# Example usage
image_path = 'path_to_your_image.jpg'  # Specify the path to your image here
main(image_path)