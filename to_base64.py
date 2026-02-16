import base64

file_path = "dataset/human_sample.wav"

with open(file_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

print(encoded)
