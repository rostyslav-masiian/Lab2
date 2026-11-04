import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def task1(matrix: np.ndarray ):
    if matrix.shape[0] != matrix.shape[1] or matrix.ndim != 2:
       return "Eigenvalues and eigenvectors cant be computed"

    vals, vects = np.linalg.eig(matrix)
    vals = np.real(vals)
    vects = np.real(vects)
    for i in range(0, len(vals)):
        if np.allclose(np.dot(matrix, vects[:, i]), vals[i] * vects[:, i]):
            print(f"{vals[i]}  - {vects[:, i]}")


print("Task 1")

test_matrix = np.random.randint(-10, 10, (3, 3)).astype(float)

print(test_matrix)
task1(test_matrix)


def task2(im_path: str):
    image_raw = imread(im_path)
    print(image_raw.shape)
    image_sum = image_raw.sum(axis=2)
    print(image_sum.shape)
    image_bw = image_sum/image_sum.max()

    pca_full = PCA().fit(image_bw)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
    target_95 = np.argmax(cum_var >= 95) + 1
    print(f"Component for dispersion of 95%: {target_95}")


    plt.plot(cum_var)
    plt.axhline(95, color='r', linestyle='--')
    plt.axvline(target_95, color='b', linestyle='--')
    plt.show()
    components = [5, 15, 25, 75, 100, 170]
    figure, axes = plt.subplots(2, 3, figsize=(10, 6))

    for ax, n in zip(axes.ravel(), components):
        pca = PCA(n_components=n)
        img_rec = pca.inverse_transform(pca.fit_transform(image_bw))
        ax.imshow(img_rec, cmap='gray')
        ax.set_title(f"Components: {n}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

print("Task 2")

task2("image.png")


def decrypt_message(encrypted_vector: np.ndarray, key_matrix: np.ndarray):
    vals, vecs = np.linalg.eig(key_matrix)
    diagonal = vecs @ np.diag(vals) @ np.linalg.inv(vecs)
    message_vec = np.linalg.inv(diagonal) @ encrypted_vector
    
    ascii_codes = np.round(np.real(message_vec)).astype(int)
    return "".join(chr(code) for code in ascii_codes)

def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_matrix, message_vector)
    return encrypted_vector


print("Task 3")
message = "Hello, World!"

key_matrix = np.random.randint(0, 256, (len(message), len(message)))
encrypted_vector = encrypt_message(message, key_matrix)
print("Encrypted:", encrypted_vector)

decrypted_message = decrypt_message(encrypted_vector, key_matrix)
print("Decrypted:", decrypted_message)