# test_svante.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from clip import clip  # ← use the local CLIP code

# ✅ Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
model.eval()

# ✅ Preprocessing for MNIST to match CLIP input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

# ✅ Load MNIST test subset (first 256 images)

mnist_full = datasets.MNIST(root="data/MNIST", train=False, download=True, transform=transform)
mnist_subset = Subset(mnist_full, range(256))
loader = DataLoader(mnist_subset, batch_size=32, shuffle=False)

# ✅ Text prompts for digits 0–9
digit_texts = [f"a photo of the digit {i}" for i in range(10)]
text_tokens = clip.tokenize(digit_texts).to(device)

# ✅ Encode text prompts once
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    all_image_features = []
    all_labels = []
    all_similarities = []

    for images, labels in loader:
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T  # Cosine similarity

        all_image_features.append(image_features.cpu())
        all_labels.append(labels)
        all_similarities.append(similarity.cpu())

# ✅ Save to file locally
torch.save({
    "image_features": torch.cat(all_image_features),
    "labels": torch.cat(all_labels),
    "text_features": text_features.cpu(),
    "similarities": torch.cat(all_similarities)
}, "mnist_clip_test_results.pt")

print("✅ Done! Saved to mnist_clip_test_results.pt")
