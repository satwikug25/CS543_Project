import os
from PIL import Image
from google import genai

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, but that's okay - user might have set env vars manually
    pass

# Read API keys from environment
keys = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY2"),
    os.getenv("GEMINI_API_KEY3"),
    os.getenv("GEMINI_API_KEY4"),
]

def test_vlm_api(image_path, test_key_index=0):
    """Test the VLM API with a single image"""
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        print("\nAvailable images in grid_piece folder:")
        if os.path.exists("grid_piece"):
            images = [f for f in os.listdir("grid_piece") if f.endswith('.png')]
            for img in images[:5]:  # Show first 5
                print(f"  - grid_piece/{img}")
            if len(images) > 5:
                print(f"  ... and {len(images) - 5} more")
        return False
    
    # Check if we have any valid keys
    valid_keys = [key for key in keys if key]
    if not valid_keys:
        print("Error: No API keys found in environment variables!")
        print("Please set at least one of: GEMINI_API_KEY, GEMINI_API_KEY2, GEMINI_API_KEY3, GEMINI_API_KEY4")
        print("\nIf you have a .env file, make sure python-dotenv is installed:")
        print("  pip install python-dotenv")
        print("  or")
        print("  poetry add python-dotenv")
        return False
    
    # Use the specified key index, or first available key
    key_to_use = None
    if test_key_index < len(keys) and keys[test_key_index]:
        key_to_use = keys[test_key_index]
    else:
        key_to_use = valid_keys[0]
    
    print(f"Testing VLM API with image: {image_path}")
    print(f"Using API key ending in: ...{key_to_use[-4:] if key_to_use else 'N/A'}")
    print("-" * 50)
    
    try:
        client = genai.Client(api_key=key_to_use)
        img = Image.open(image_path)
        
        print("Sending request to Gemini API...")
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                img,
                (
                    "You will be given a top-down photo of a square. "
                    "Your task is to check if the square has a circular object and classify into one of three categories: "
                    "empty, object, object-black. "
                    "Respond with exactly one of these labels and nothing else."
                )
            ],
        )
        
        label = response.text.strip().lower()
        print(f"✓ API call successful!")
        print(f"Response: {label}")
        print("-" * 50)
        
        # Validate response
        valid_labels = ['empty', 'object', 'object-black']
        if label in valid_labels:
            print(f"✓ Response is valid: '{label}'")
            return True
        else:
            print(f"⚠ Response is not one of the expected labels: {valid_labels}")
            print(f"  Got: '{label}'")
            return False
            
    except Exception as e:
        print(f"✗ API call failed!")
        print(f"Error: {e}")
        print("-" * 50)
        
        # Try next available key if this one failed
        if test_key_index < len(valid_keys) - 1:
            print(f"\nTrying next available API key...")
            return test_vlm_api(image_path, test_key_index + 1)
        
        return False

if __name__ == "__main__":
    # Test with the first available grid piece image
    test_image = "grid_piece/piece_r0_c0.png"
    
    # If that doesn't exist, try to find any image
    if not os.path.exists(test_image):
        if os.path.exists("grid_piece"):
            images = [f"grid_piece/{f}" for f in os.listdir("grid_piece") if f.endswith('.png')]
            if images:
                test_image = images[0]
            else:
                print("No images found in grid_piece folder!")
                print("Please run the pipeline first to generate grid piece images, or specify an image path.")
                exit(1)
        else:
            print("grid_piece folder not found!")
            exit(1)
    
    success = test_vlm_api(test_image)
    
    if success:
        print("\n✓ VLM API is working correctly!")
    else:
        print("\n✗ VLM API test failed. Please check your API keys and network connection.")

