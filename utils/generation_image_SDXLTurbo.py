import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from diffusers import StableDiffusionPipeline
from modelscope import AutoPipelineForText2Image
import torch
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

model_id = "/data0/fayi/sdxl"
pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.safety_checker = None
pipe = pipe.to("cuda")

def sanitize_label(label):
    """Remove invalid characters from labels (e.g., newlines, special chars)."""
    return label.replace("/", "_").replace("\n", "").replace("\"", "").strip()


def generation(labels, img_dir, num_images=3, batch_size=4):
    all_tasks = []
    for label in labels:
        sanitized_label = sanitize_label(label)
        category_dir = os.path.join(img_dir, sanitized_label)
        os.makedirs(category_dir, exist_ok=True)

        for i in range(num_images):
            path = os.path.join(category_dir, f"{sanitized_label}_{i + 1}.png")
            if os.path.exists(path):
                continue

            prompt = f"{label}"
            seed = hash(f"{sanitized_label}_{i}") % (2 ** 32)
            all_tasks.append((prompt, seed, path, sanitized_label))

    for i in tqdm(range(0, len(all_tasks), batch_size), desc="Generating images in batches"):
        batch = all_tasks[i:i+batch_size]
        if not batch:
            continue

        for j, (prompt, seed, path, _) in enumerate(batch):
            generator = torch.manual_seed(seed)
            images = pipe(prompt, generator=generator, num_inference_steps=1, guidance_scale=0.0).images

            if images and len(images) > 0:
                images[0].save(path)
                print(f"Generated and saved {path}")
            else:
                print(f"Failed to generate image for {prompt}")


if __name__ == '__main__':
    positive_samples_path = PROJECT_ROOT / 'OODNegMining' / 'CXR' / 'positive' / 'CXR.txt'
    low_similarity_neg_samples_path = PROJECT_ROOT / 'OODNegMining' / 'CXR' / 'negative' / 'CXR_neg.txt'

    with open(positive_samples_path, 'r') as f:
        positive_samples = f.readlines()

    with open(low_similarity_neg_samples_path, 'r') as f:
        low_similarity_neg_samples = f.readlines()

    combined_samples =  positive_samples + low_similarity_neg_samples
    generation(combined_samples, img_dir='/data0/fayi/generation_image_xl', num_images=1, batch_size=1)
