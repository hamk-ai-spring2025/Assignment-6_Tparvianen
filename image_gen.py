import os
import argparse
import replicate
import requests
from datetime import datetime

# Load API token
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN not set as environment variable.")

# Initialize Replicate client
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Argument parser for CLI
parser = argparse.ArgumentParser(description="AI Image Generator using Replicate SDXL")
parser.add_argument("--prompt", required=True, help="Prompt for image generation")
parser.add_argument("--negative", help="Negative prompt (what to avoid)")
parser.add_argument("--seed", type=int, help="Seed for reproducibility")
parser.add_argument("--aspect", default="square", choices=["square", "landscape", "portrait"], help="Aspect ratio")
parser.add_argument("--n", type=int, default=1, help="Number of images to generate")

args = parser.parse_args()

# Aspect ratio mapping
aspect_map = {
    "square": "1024x1024",
    "landscape": "1280x768",
    "portrait": "768x1280"
}

# Parse width and height
width, height = map(int, aspect_map[args.aspect].split("x"))

# Get latest version of SDXL model
model = client.models.get("stability-ai/sdxl")
version = model.versions.list()[0]  # latest version

print("⏳ Generating image(s)...")

# Run the prediction
prediction = client.predictions.create(
    version=version.id,
    input={
        "prompt": args.prompt,
        "negative_prompt": args.negative or "",
        "width": width,
        "height": height,
        "num_outputs": args.n,
        "guidance_scale": 7.5,
        "num_inference_steps": 30,
        "seed": args.seed if args.seed is not None else 42
    },
    wait=True  # <-- This makes it wait for the image to finish
)

# Get image URLs
output = prediction.output
print("DEBUG status:", prediction.status)
print("DEBUG raw prediction:", prediction)

if not output:
    print("❌ No images were generated. Something went wrong.")
else:
    # Download and save each image
    for i, url in enumerate(output):
        filename = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}.png"
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)
        print(f"✅ Image {i+1} saved as {filename}")
        print(f"🌐 URL: {url}")

