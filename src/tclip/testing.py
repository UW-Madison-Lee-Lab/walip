import json

path = "../../data/coco/annotations/captions_val2017.json"
with open(path) as f:

    data = json.load(f)
    print(data["images"])
    print(data["annotations"])
