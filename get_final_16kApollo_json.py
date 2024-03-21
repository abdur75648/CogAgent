import os, re, json
from tqdm import tqdm

data_dir = "16k-Apollo-MLLM-data"

# rounded_data.json inside data_dor
file_name = os.path.join(data_dir, "rounded_data.json")
with open(file_name, 'r') as file:
    dataset = json.load(file)

print("Dataset size: ", len(dataset))

for data in tqdm(dataset):
    image_path = data['imagePath']
    image_path = os.path.join(data_dir, image_path)
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
print("Image files checked!\n")

for data in tqdm(dataset):
    bbox = re.findall(r"\[.*?\]", data["Answer"])
    if len(bbox) != 1:
        print(f"Invalid bounding box: {data['Answer']}")
print("Answer's Bounding boxes checked!\n")

invalid_questions = 0
for data in tqdm(dataset):
    bbox = re.findall(r"\[.*?\]", data["Question"])
    if len(bbox) != 0:
        invalid_questions += 1
if invalid_questions > 0:
    print(f"Found {invalid_questions} invalid questions!")
    print("Removing bounding boxes from such Questions...")
    for data in tqdm(dataset):
        data["Question"] = re.sub(r"\[.*?\]", "", data["Question"])
    print("Bounding boxes removed from Questions!")
    invalid_questions = 0
    for data in tqdm(dataset):
        bbox = re.findall(r"\[.*?\]", data["Question"])
        if len(bbox) != 0:
            print(f"Invalid question: {data['Question']}")
            invalid_questions += 1
    if invalid_questions > 0:
        print(f"Still found {invalid_questions} invalid questions!")
    else:
        print("Questions checked!\n")
else:
    print("Questions checked!\n")

print("Done!")

# Save the updated dataset to a new file named "final_dataset.json"
with open(os.path.join(data_dir, "final_dataset.json"), 'w') as file:
    json.dump(dataset, file, indent=4)