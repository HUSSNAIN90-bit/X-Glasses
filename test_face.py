from app.services.face_recognition import (
    extract_face_embedding,
)


image_path: str = "test-face.jpg"

embedding = extract_face_embedding(
    image_path
)

if embedding is None:
    print("No face detected.")
else:
    print("FACE OK")
    print("Embedding shape:", embedding.shape)
    print("Embedding length:", len(embedding))
    print("First values:", embedding[:5])