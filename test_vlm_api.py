import os
from PIL import Image
from google import genai

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

api_key_list = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY2"),
    os.getenv("GEMINI_API_KEY3"),
    os.getenv("GEMINI_API_KEY4"),
]

def run_api_test(img_path, key_idx=0):
    
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found!")
        print("\nAvailable images in grid_piece folder:")
        if os.path.exists("grid_piece"):
            img_files = [f for f in os.listdir("grid_piece") if f.endswith('.png')]
            for img in img_files[:5]:
                print(f"  - grid_piece/{img}")
            if len(img_files) > 5:
                print(f"  ... and {len(img_files) - 5} more")
        return False
    
    working_keys = [k for k in api_key_list if k]
    if not working_keys:
        print("Error: No API keys found in environment variables!")
        print("Please set at least one of: GEMINI_API_KEY, GEMINI_API_KEY2, GEMINI_API_KEY3, GEMINI_API_KEY4")
        print("\nIf you have a .env file, make sure python-dotenv is installed:")
        print("  pip install python-dotenv")
        print("  or")
        print("  poetry add python-dotenv")
        return False
    
    selected_key = None
    if key_idx < len(api_key_list) and api_key_list[key_idx]:
        selected_key = api_key_list[key_idx]
    else:
        selected_key = working_keys[0]
    
    print(f"Testing VLM API with image: {img_path}")
    print(f"Using API key ending in: ...{selected_key[-4:] if selected_key else 'N/A'}")
    print("-" * 50)
    
    try:
        api_client = genai.Client(api_key=selected_key)
        test_img = Image.open(img_path)
        
        print("Sending request to Gemini API...")
        api_response = api_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                test_img,
                (
                    "You will be given a top-down photo of a square. "
                    "Your task is to check if the square has a circular object and classify into one of three categories: "
                    "empty, object, object-black. "
                    "Respond with exactly one of these labels and nothing else."
                )
            ],
        )
        
        result = api_response.text.strip().lower()
        print(f"✓ API call successful!")
        print(f"Response: {result}")
        print("-" * 50)
        
        expected_labels = ['empty', 'object', 'object-black']
        if result in expected_labels:
            print(f"✓ Response is valid: '{result}'")
            return True
        else:
            print(f"⚠ Response is not one of the expected labels: {expected_labels}")
            print(f"  Got: '{result}'")
            return False
            
    except Exception as err:
        print(f"✗ API call failed!")
        print(f"Error: {err}")
        print("-" * 50)
        
        if key_idx < len(working_keys) - 1:
            print(f"\nTrying next available API key...")
            return run_api_test(img_path, key_idx + 1)
        
        return False

if __name__ == "__main__":
    sample_img = "grid_piece/piece_r0_c0.png"
    
    if not os.path.exists(sample_img):
        if os.path.exists("grid_piece"):
            available_imgs = [f"grid_piece/{f}" for f in os.listdir("grid_piece") if f.endswith('.png')]
            if available_imgs:
                sample_img = available_imgs[0]
            else:
                print("No images found in grid_piece folder!")
                print("Please run the pipeline first to generate grid piece images, or specify an image path.")
                exit(1)
        else:
            print("grid_piece folder not found!")
            exit(1)
    
    test_passed = run_api_test(sample_img)
    
    if test_passed:
        print("\n✓ VLM API is working correctly!")
    else:
        print("\n✗ VLM API test failed. Please check your API keys and network connection.")
