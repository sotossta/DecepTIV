import importlib
import cv2
from PIL import Image
from io import BytesIO
import base64

def load_generator_from_config(config):

    generator_name = config['generator']
    # Dynamically import the generator class
    generator_module = importlib.import_module(f".{generator_name}", package="video_generators")
    # Get the class from the module
    generator_class = getattr(generator_module,f"{generator_name}_Generator")
    # Instantiate the generator
    generator = generator_class(config)
    return generator


def pil_image_to_url(pil_image, mime_type="image/png"):
    """
    Convert a PIL image to a Base64-encoded data URL.
    
    Args:
        pil_image (PIL.Image.Image): The input PIL image.
        mime_type (str): The MIME type of the image (default: 'image/png').
        
    Returns:
        str: A Base64-encoded data URL of the image.
    """
    buffer = BytesIO()
    pil_image.save(buffer, format=mime_type.split("/")[-1].upper())
    buffer.seek(0)
    base64_encoded_data = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"

def extract_first_frame(video_path):
    """
    Extract the first frame of a video as a PIL image.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        PIL.Image.Image: The first frame of the video as a PIL image.
        None: If the video cannot be read or processed.
    """
    # Open the video file using OpenCV
    video_capture = cv2.VideoCapture(video_path)
    # Check if the video file is opened successfully
    if not video_capture.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return None
    # Read the first frame
    success, frame = video_capture.read()
    # Release the video capture object
    video_capture.release()
    if not success:
        print("Error: Cannot read the first frame of the video.")
        return None
    # Convert the frame (which is in BGR format) to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the frame to a PIL image
    pil_image = Image.fromarray(frame_rgb)
    return pil_image