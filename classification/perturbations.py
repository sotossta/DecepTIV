import cv2
import numpy as np
import random
#import ffmpeg
import math
import os
import string

def ycbcr2bgr(frame_ycbcr):
     frame_ycbcr = frame_ycbcr.astype(np.float32)
     # to [0, 1]
     frame_ycbcr[:, :, 0] = (frame_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
     # to [0, 1]
     frame_ycbcr[:, :, 1:] = (frame_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
     frame_ycrcb = frame_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
     frame_bgr = cv2.cvtColor(frame_ycrcb, cv2.COLOR_YCR_CB2BGR)

     return frame_bgr
 
def bgr2ycbcr(frame_bgr):
     frame_bgr = frame_bgr.astype(np.float32)
     frame_ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCR_CB)
     frame_ycbcr = frame_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
     # to [16/255, 235/255]
     frame_ycbcr[:, :, 0] = (frame_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
     # to [16/255, 240/255]
     frame_ycbcr[:, :, 1:] = (frame_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0

     return frame_ycbcr
 
    
# Function to generate random text of certain length
def generate_random_text(length):  
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Function to generate a random color 
def generate_random_color():
    # Randomly generate a color in BGR format
    return (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1))

# Function to create a Deepfakes logo
def create_logo(text="Deepfakes", font_scale=2, thickness=2, width=250, height=50):
    # Create a black image to place the text
    logo = 1 * np.ones((height, width, 3), dtype=np.uint8)

    # Define the font and calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Calculate the center position for the text
    text_x = (logo.shape[1] - text_size[0]) // 2
    text_y = (logo.shape[0] + text_size[1]) // 2

    # Add the text to the logo
    cv2.putText(logo, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    return logo

# Function to generate fixed shapes
def generate_fixed_shapes(num_shapes, frame_size):
    height, width, _ = frame_size
    shapes = []

    for _ in range(num_shapes):
        shape_type = random.choice(["circle", "rectangle", "line"])
        color = tuple(random.randint(0, 255) for _ in range(3))  # Random color (BGR)
        thickness = random.randint(1, 4)  # Random thickness

        if shape_type == "circle":
            center = (random.randint(0, width), random.randint(0, height))
            radius = random.randint(10, min(width, height) // 10)
            shapes.append(("circle", center, radius, color, thickness))

        elif shape_type == "rectangle":
            x1, y1 = random.randint(0, width - 50), random.randint(0, height - 50)
            x2, y2 = x1 + random.randint(10, 50), y1 + random.randint(10, 50)
            shapes.append(("rectangle", (x1, y1), (x2, y2), color, thickness))

        elif shape_type == "line":
            pt1 = (random.randint(0, width), random.randint(0, height))
            pt2 = (random.randint(0, width), random.randint(0, height))
            shapes.append(("line", pt1, pt2, color, thickness))

    return shapes
 
#--------- Augmenters --------------------

# Gaussian Blur
def apply_gaussian_blur(frame, param):
    return cv2.GaussianBlur(frame, (param, param), param * 1.0 / 6)

# Gaussian noise
def add_gaussian_noise(frame, param):
    
    ycbcr = bgr2ycbcr(frame) / 255
    size_a = ycbcr.shape
    b = (ycbcr + math.sqrt(param) *
         np.random.randn(size_a[0], size_a[1], size_a[2])) * 255
    b = ycbcr2bgr(b)
    frame = np.clip(b, 0, 255).astype(np.uint8)
    return frame

# Constrast adjustment
def apply_color_contrast(frame, param):
    frame = frame.astype(np.float32) * param
    frame = frame.astype(np.uint8)
    return frame

# Brightness adjustment
def apply_color_brightness(frame, param):

    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.int16) + param, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v.astype(np.uint8)))
    frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return frame

# Saturation adjustment
def apply_color_saturation(frame, param):
    ycbcr = bgr2ycbcr(frame)
    ycbcr[:, :, 1] = 0.5 + (ycbcr[:, :, 1] - 0.5) * param
    ycbcr[:, :, 2] = 0.5 + (ycbcr[:, :, 2] - 0.5) * param
    frame = ycbcr2bgr(ycbcr).astype(np.uint8)
    return frame

# JPEG compression
def jpeg_compression(frame, param):
    h, w, _ = frame.shape
    s_h = h // param
    s_w = w // param
    frame = cv2.resize(frame, (s_w, s_h))
    frame = cv2.resize(frame, (w, h))
    return frame

# Video h.264 compression
def video_compression(vid_in, vid_out, param):
    cmd = f"ffmpeg -i {vid_in} -vcodec libx264 -preset medium -crf {param} -y {vid_out}"
    os.system(cmd)
    return

# Rotate Image
def rotate_image(frame, angle):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, matrix, (w, h))


# Change Resolution (Downscale & Upscale)
def change_resolution(frame, scale):
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)

#--------- Distractors --------------------

# Block distractors
def add_block_distractors(frame, param):
    width = 8
    block = np.ones((width, width, 3)).astype(int) * 128
    param = min(frame.shape[0], frame.shape[1]) // 256 * param
    for i in range(param):
        r_w = random.randint(0, frame.shape[1] - 1 - width)
        r_h = random.randint(0, frame.shape[0] - 1 - width)
        frame[r_h:r_h + width, r_w:r_w + width, :] = block

    return frame


# Add random text to frame
def add_text_distractor(frame, param, text, position, text_color, font_scale=1, thickness=2):
    # Define the font and calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Use the position passed to place the text
    text_x, text_y = position

    # Add the static text on the frame
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

    return frame

# Add logo distractor to frame
def add_logo_distractor(frame,param, logo, x_pos, y_pos):
    # Get the size of the logo and the frame
    frame_height, frame_width = frame.shape[:2]
    logo_height, logo_width = logo.shape[:2]

    # Ensure the logo fits within the frame (crop if necessary)
    if x_pos + logo_width > frame_width:
        logo_width = frame_width - x_pos
    if y_pos + logo_height > frame_height:
        logo_height = frame_height - y_pos

    # Overlay the logo on the frame (without transparency)
    frame[y_pos:y_pos+logo_height, x_pos:x_pos+logo_width] = logo[:logo_height, :logo_width]

    return frame

def add_shape_distractors(frame,param, shapes):
    
    for shape in shapes:
        if shape[0] == "circle":
            _, center, radius, color, thickness = shape
            cv2.circle(frame, center, radius, color, thickness)

        elif shape[0] == "rectangle":
            _, pt1, pt2, color, thickness = shape
            cv2.rectangle(frame, pt1, pt2, color, thickness)

        elif shape[0] == "line":
            _, pt1, pt2, color, thickness = shape
            cv2.line(frame, pt1, pt2, color, thickness)

    return frame

#----------- Temporal Tampering --------- #

# Drop random frames
def drop_random_frames(frames, drop_fraction=0.2):
    n = len(frames)
    num_to_drop = int(n * drop_fraction)
    keep_indices = sorted(random.sample(range(n), n - num_to_drop))
    return [frames[i] for i in keep_indices]

#Frame rate change
def change_frame_rate(frames, speed_factor=2):
    if speed_factor >= 1:
        return frames[::int(speed_factor)]
    else:
        # Slow down: duplicate frames
        output = []
        dup_count = int(1 / speed_factor)
        for frame in frames:
            output.extend([frame] * dup_count)
        return output



