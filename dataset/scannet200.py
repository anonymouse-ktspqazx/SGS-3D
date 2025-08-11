# ScanNet200 dataset configuration
# This is a simplified version for anonymous review

INSTANCE_CAT_SCANNET_200 = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat',
    'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
    'person', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop',
    'bicycle', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing machine', 'mirror', 'copier', 'basket',
    'sofa chair', 'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper towel dispenser', 'oven', 'blinds',
    'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container', 'wardrobe',
    'soap dispenser', 'telephone', 'bucket', 'clock', 'stand', 'computer', 'bottle', 'board', 'cup', 'paper',
    'keyboard', 'electronics', 'shoe', 'trash can', 'printer', 'speaker', 'microwave', 'mat', 'mousepad', 'water bottle',
    'whiteboard', 'purse', 'wallet', 'bed', 'monitor', 'curtain', 'book', 'armchair', 'coffee table', 'box',
    'lounge chair', 'dining table', 'console table', 'plant', 'ceiling', 'bathtub', 'end table', 'dining chair', 'keyboard piano', 'dresser',
    'coffee maker', 'toilet paper', 'stove', 'drawer', 'cup', 'ceiling fan', 'trash bin', 'vacuum cleaner', 'dishwasher', 'range hood',
    'dustpan', 'hair dryer', 'water heater', 'paper cutter', 'treadmill', 'file', 'ball', 'tennis racket', 'briefcase', 'humidifier',
    'toothbrush', 'hair brush', 'bananas', 'apple', 'orange', 'potted plant', 'mouse', 'toilet seat cover dispenser', 'kleenex', 'soap bottle',
    'keyboard', 'phone', 'alarm clock', 'music stand', 'projector', 'divider', 'laundry basket', 'bathroom vanity', 'bulletin board', 'bottle',
    'jug', 'magazine', 'light', 'monitor', 'radiator', 'glass', 'desk organizer', 'soap', 'hand towel', 'machine',
    'mat', 'bicycle', 'kneeling chair', 'basket', 'music book', 'sign', 'tissue paper', 'person', 'apple', 'knife block',
    'vacuum cleaner', 'cardboard', 'scale', 'tissue', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', 'projector',
    'closet door', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack', 'broom', 'guitar', 'pillow',
    'shoe', 'meter', 'dustpan', 'oven', 'tray', 'range hood', 'microwave', 'pot', 'animal', 'bicycle',
    'hair dryer'
]

# Simplified class mapping for anonymous review
SCANNET200_CLASS_MAPPING = {
    'furniture': ['chair', 'table', 'sofa', 'bed', 'desk', 'cabinet', 'shelf', 'dresser'],
    'appliances': ['refrigerator', 'microwave', 'oven', 'washing machine', 'dishwasher'],
    'electronics': ['television', 'computer', 'laptop', 'monitor', 'phone', 'speaker'],
    'containers': ['box', 'bag', 'basket', 'bin', 'bottle', 'cup'],
    'structural': ['wall', 'floor', 'ceiling', 'door', 'window'],
    'lighting': ['lamp', 'light'],
    'textiles': ['curtain', 'pillow', 'towel', 'clothes', 'mat'],
    'bathroom': ['toilet', 'sink', 'bathtub', 'shower'],
    'office': ['whiteboard', 'blackboard', 'printer', 'copier', 'file cabinet'],
    'other': ['person', 'plant', 'picture', 'mirror', 'clock']
}

def get_scannet200_classes():
    """Return ScanNet200 class names"""
    return INSTANCE_CAT_SCANNET_200

def get_class_mapping():
    """Return simplified class mapping"""
    return SCANNET200_CLASS_MAPPING

def map_class_to_category(class_name):
    """Map individual class to broader category"""
    for category, classes in SCANNET200_CLASS_MAPPING.items():
        if class_name.lower() in [c.lower() for c in classes]:
            return category
    return 'other'
