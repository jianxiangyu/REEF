import json

def relation_description(relation):
    with open('./relation_info.json', 'r') as f:
        relation_dict = json.load(f)
        assert relation in relation_dict["relation_description"]
        return relation_dict["relation_description"][relation]