from PIL import Image
from google import genai
import os
import json

# Read your four keys from environment
keys = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY2"),
    os.getenv("GEMINI_API_KEY3"),
    os.getenv("GEMINI_API_KEY4"),
]
num_keys = len(keys)

# Path to the grayscale folder
gray_scale_folder = "grid_piece"

# Loop bounds and stop condition
max_r, max_c = 7, 7
stop_r, stop_c = 7, 7

results = []
current_key_index = 0  # start with the first key

for r in range(max_r + 1):
    for c in range(max_c + 1):
        filename = f"piece_r{r}_c{c}.png"
        image_path = os.path.join(gray_scale_folder, filename)
        if not os.path.isfile(image_path):
            print(f"[skip] {filename} not found.")
            continue

        print(f"Processing {filename} (r={r}, c={c})")
        label = "error"

        # Try up to num_keys different keys only on error
        attempts = 0
        while attempts < num_keys:
            key = keys[current_key_index]

            # Skip missing keys
            if not key:
                print(f"[skip key] missing key at index {current_key_index}")
                current_key_index = (current_key_index + 1) % num_keys
                attempts += 1
                continue

            try:
                client = genai.Client(api_key=key)
                img = Image.open(image_path)
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=[
                        img,
                        (
                            "You will be given a top-down photo of a square. "
                            "Your task is to check if the square has a circular object and classify into one of three categories: "
                            "empty, object, object-golden. "
                            "Respond with exactly one of these labels and nothing else."
                        )
                    ],
                )
                label = response.text.strip().lower()
                print(f"→ {label}  (used key ending …{key[-4:]})")
                break

            except Exception as e:
                print(f"[error with key {key}] {e}")
                current_key_index = (current_key_index + 1) % num_keys
                attempts += 1

        if attempts == num_keys and label == "error":
            print(f"[failed] {filename}: all keys errored out.")

        results.append((r, c, label))

        # Stop if we've hit the designated cell
        if r == stop_r and c == stop_c:
            break
    else:
        continue
    break

# --- append to detection.json ---
detection_file = "detection.json"

# Load existing entries
if os.path.exists(detection_file):
    try:
        with open(detection_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []
else:
    data = []

# Assign a simple incremental ID
new_id = len(data) + 1

# Build this run’s entry with ID instead of timestamp
run_entry = {
    "id": new_id,
    "results": [
        {"r": r, "c": c, "label": label}
        for r, c, label in results
    ]
}

data.append(run_entry)

with open(detection_file, "w") as f:
    json.dump(data, f, indent=2)

# Finally, print the summary table
print("\nClassification Results:\n")
print("r | c | label")
print("---|---|---")
for r, c, label in results:
    print(f"{r} | {c} | {label}")

print(f"\nDone. Detection saved to {detection_file} (entry id: {new_id})")
