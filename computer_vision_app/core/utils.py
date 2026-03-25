import cv2
import numpy as np
import base64

def decode_image(image_b64):
    """Converts a Base64 image to an OpenCV frame (numpy array)."""
    try:
        if not image_b64.startswith("data:image/"):
             return None
        
        # Separates the header ('data:image/jpeg;base64,') from the encoded string
        _, encoded = image_b64.split(",", 1)
        
        # Converts from b64 to byte array
        img_bytes = base64.b64decode(encoded)
        
        # Converts to numpy array and decodes via OpenCV (BGR format)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def encode_image(frame, quality=50):
    """Converts an OpenCV frame (numpy array) to a Base64 string with dataURI prefix."""
    try:
        # Encodes to JPEG format in memory
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        # Converts to base64 string
        b64_str = base64.b64encode(buffer).decode()
        
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception as e:
        print(f"Error encoding image: {e}")
        return ""
