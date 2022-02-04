import json
import pandas as pd

ANNOTATION_PATH = "../../dataset/coco/annotations/captions_val2014.json"
INSTANCE_FILE_PATH = "../../dataset/coco/annotations/instances_val2014.json"

# Saves a , separated csv to this location:
SAVE_PATH = "../../dataset/coco/captions/en/processed_captions_val2014.csv"

with open(ANNOTATION_PATH) as f:
    data = json.load(f)

with open(INSTANCE_FILE_PATH) as f:
    data_instance = json.load(f)


category_to_name = {} # Mapping from category to name
original_id_to_pytorch_id = {} # original ids (can be between 1 and 90) to pytorch 0-79 ids

for i, category in enumerate(data_instance["categories"]):
    category_to_name[i] = category['name']
    original_id_to_pytorch_id[category['id']] = i

# Get labels for each image
# each entry looks like
#   'segment': ...
#   'area': 702.1057499999998,
#   'iscrowd': 0,
#   'image_id': 289343,
#   'bbox': [473.07, 395.93, 38.65, 28.67],
#   'category_id': 18,
#   'id': 1768
image_info = {} # maps image_id to [0,0,1,0,...,1,...] <-- 1 if the class is present in the image
for box in data_instance["annotations"]:
    idx = original_id_to_pytorch_id[box['category_id']]
    if box['image_id'] not in image_info:
        image_info[box['image_id']] = []
    image_info[box['image_id']].append(str(idx))

for image_id in image_info:
    image_info[image_id] = " ".join(list(set(image_info[image_id])))

# Go through the caption annotations, and add labels to them
for i in range(len(data["annotations"])):
    image_id = data["annotations"][i]["image_id"]
    try:
        data["annotations"][i]["labels"] = image_info[image_id]
    except KeyError:
        print("Skipping image id {} which has no coco objects".format(image_id))


df = pd.DataFrame(data["annotations"])
df = df[df["labels"].notna()]
df.to_csv(SAVE_PATH, index=False)

