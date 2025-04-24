import numpy as np
import torch
from torchvision import datasets, transforms
import tester
def main():
    N = 7000

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(3)])
    mnist = datasets.MNIST(root = './data', download = True, train = True, transform=transform)

    image_data = torch.load("image_embeddings.pt")
    enc_images = image_data["image"]  # shape (N, D)
    labels = image_data["labels"]  # list of ints, length N

    text_data = torch.load("text_embeddings.pt")
    enc_prompts = text_data["prompt"]  # shape (10, D)


    similarity = enc_images @ enc_prompts.T

    prediction = similarity.argmax(dim = 1).cpu().numpy()

    labels = [mnist[i][1] for i in range(N)]
    labels = np.array(labels)

    accuracy = (prediction == labels).mean()
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()