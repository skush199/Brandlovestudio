import openai
from google.cloud import vision
from google.oauth2 import service_account
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Set up OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure this matches the variable name in .env

# Set up Google Vision API credentials from the environment variable
google_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')  # Path to your Google service account JSON
credentials = service_account.Credentials.from_service_account_file(google_credentials_path)
client = vision.ImageAnnotatorClient(credentials=credentials)

# Function to process the image and identify objects using Google Vision API
def process_image_with_vision(image_path):
    with open(image_path, 'rb') as img_file:
        content = img_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)

    if response.error.message:
        raise Exception(f"Error with Google Vision API: {response.error.message}")

    labels = [label.description for label in response.label_annotations]
    return labels

# Function to send the detected objects to OpenAI for generating detailed descriptions
def generate_description_with_openai(labels):
    prompt = f"Please describe the following objects in detail as if you are explaining them in a sentence: {', '.join(labels)}."
    
    response = openai.Completion.create(
        model="gpt-4.1",  # Use GPT-4 or another suitable model
        prompt=prompt,
        max_tokens=50,
        temperature=0.7
    )
    
    description = response['choices'][0]['text'].strip()
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