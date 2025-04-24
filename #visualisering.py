#visualisering
# evaluate_svante.py

import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ✅ Load the saved results
data = torch.load("mnist_clip_test_results.pt")
image_features = data["image_features"]
text_features = data["text_features"]
labels = data["labels"]
similarities = data["similarities"]

# ✅ Predict the most likely digit by finding the max similarity
predicted_labels = similarities.argmax(dim=1)
accuracy = (predicted_labels == labels).float().mean()

print(f"✅ CLIP zero-shot accuracy on MNIST: {accuracy.item():.2%}")

# ✅ Optional: Show some mismatches
mismatches = (predicted_labels != labels).nonzero().squeeze()
if mismatches.numel() > 0:
    print(f"\n⚠️ First few mismatches:")
    for i in mismatches[:5]:
        print(f"True: {labels[i].item()}, Predicted: {predicted_labels[i].item()}")

# ✅ Optional: t-SNE Visualization of Image Embeddings
print("\n🔄 Running t-SNE on image embeddings... (this may take 10-30 sec)")
tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(image_features)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", alpha=0.7, s=15)
plt.colorbar(scatter, ticks=range(10), label="Digit Label")
plt.title("t-SNE of CLIP Image Embeddings (MNIST Subset)")
plt.tight_layout()
plt.show()
