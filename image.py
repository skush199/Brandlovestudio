# import openai
# from openai import OpenAI
# from google.cloud import vision
# from google.oauth2 import service_account
# from dotenv import load_dotenv
# import os

# # Load environment variables from the .env file
# load_dotenv()

# # Set up OpenAI client using the new SDK
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Set up Google Vision API credentials from the environment variable
# google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# credentials = service_account.Credentials.from_service_account_file(
#     google_credentials_path
# )
# vision_client = vision.ImageAnnotatorClient(credentials=credentials)


# # Function to process the image and identify objects using Google Vision API
# def process_image_with_vision(image_path):
#     with open(image_path, "rb") as img_file:
#         content = img_file.read()

#     image = vision.Image(content=content)

#     response = vision_client.label_detection(image=image)

#     if response.error.message:
#         raise Exception(f"Error with Google Vision API: {response.error.message}")

#     labels = [label.description for label in response.label_annotations]
#     return labels


# # Function to send the detected objects to OpenAI for generating detailed descriptions
# def generate_description_with_openai(labels):
#     prompt = f"Please describe the following objects in detail as if you are explaining them in a sentence: {', '.join(labels)}."

#     # Updated API call using the new OpenAI SDK
#     response = client.chat.completions.create(
#         model="gpt-4.1",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt},
#         ],
#         max_tokens=200,
#         temperature=0.7,
#     )

#     description = response.choices[0].message.content.strip()
#     return description


# # Main function to run the image processing and description generation
# def main(image_path):
#     # Step 1: Process the image to detect objects using Google Vision API
#     labels = process_image_with_vision(image_path)

#     # Step 2: Generate a detailed description using OpenAI based on the detected labels
#     description = generate_description_with_openai(labels)

#     # Step 3: Print out the detected labels and the generated description
#     print("Detected Objects:", labels)
#     print("Generated Description:", description)


# # Example usage
# image_path = "/Users/vishruth/Downloads/istockphoto-2153823097-2048x2048.jpg"  # Specify the path to your image here
# main(image_path)