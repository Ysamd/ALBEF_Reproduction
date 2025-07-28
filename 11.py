import os
import json
import xml.etree.ElementTree as ET

def get_sentence_data(fn):
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence' : ' '.join(words), 'phrases' : []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index' : index,
                                             'phrase' : phrase,
                                             'phrase_id' : p_id,
                                             'phrase_type' : p_type})

        annotations.append(sentence_data)

    return annotations

def parse_all_sentences(sentence_dir):
    """
    解析所有句子文件，合并为图片对应的多条caption列表
    """
    all_captions = {}
    for fname in os.listdir(sentence_dir):
        if not fname.endswith('.txt'):
            continue
        image_id = fname.replace('.txt', '')
        full_path = os.path.join(sentence_dir, fname)
        sentence_data = get_sentence_data(full_path)
        # 把每条sentence的纯文本加入caption列表
        captions = [item['sentence'] for item in sentence_data]
        all_captions[image_id] = captions
    return all_captions

def create_json_from_captions(captions_dict, output_path):
    """
    根据captions字典生成json文件
    """
    dataset = []
    for img, caps in captions_dict.items():
        dataset.append({
            'image': img,
            'caption': caps
        })
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Saved dataset json to {output_path}")

if __name__ == "__main__":
    # 你的句子文件夹路径，例：
    sentence_folder = "/home/alsoyyy/PycharmProjects/ALBEF/ALBEF/data/flickr30k/annotations/Sentences"
    output_json = "/home/alsoyyy/PycharmProjects/ALBEF/ALBEF/data/flickr30k/annotations/dataset_flickr30k.json"

    captions_dict = parse_all_sentences(sentence_folder)
    create_json_from_captions(captions_dict, output_json)
