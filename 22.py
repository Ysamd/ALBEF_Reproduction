import json

def add_image_id_to_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for item in data:
        image_path = item['image']  # 如 "flickr30k_images/36979.jpg"
        filename = image_path.split('/')[-1]
        image_id = int(filename.split('.')[0])
        item['image_id'] = image_id

    with open(file_path, 'w') as f:
        json.dump(data, f)

# 举例使用
add_image_id_to_json('/home/alsoyyy/PycharmProjects/ALBEF/ALBEF/data/flickr30k/annotations/train.json')
add_image_id_to_json('/home/alsoyyy/PycharmProjects/ALBEF/ALBEF/data/flickr30k/annotations/val.json')
add_image_id_to_json('/home/alsoyyy/PycharmProjects/ALBEF/ALBEF/data/flickr30k/annotations/test.json')
