from intervaltree import Interval, IntervalTree


def most_frequent(height_list):
    counter = 0
    num = height_list[0]

    for i in height_list:
        curr_frequency = height_list.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


def calculate_iou(ocr_text, checkbox):
    x1_box1, y1_box1, x2_box1, y2_box1 = ocr_text
    x1_box2, y1_box2, x2_box2, y2_box2 = checkbox

    intersection_width = min(x2_box1, x2_box2) - max(x1_box1, x1_box2)
    intersection_height = min(y2_box1, y2_box2) - max(y1_box1, y1_box2)

    if intersection_width <= 0 or intersection_height <= 0:
        intersection_area = 0
    else:
        intersection_area = intersection_width * intersection_height

    area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area

    return iou


def checkbox_in_ocr_region(checkbox_coords, ocr_coords_list):
    for ocr_coords in ocr_coords_list:
        if calculate_iou(ocr_coords, checkbox_coords) > 0.2:
            return True

    return False


def build_interval_trees(boxes):
    tree_x = IntervalTree()
    tree_y = IntervalTree()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        tree_x.add(Interval(x1, x2, i))
        tree_y.add(Interval(y1, y2, i))

    return tree_x, tree_y


def is_surrounded(tree_x, tree_y, target_box):
    x1, y1, x2, y2 = target_box

    overlaps_x = tree_x[x1:x2]
    overlaps_y = tree_y[y1:y2]

    for ov_x in overlaps_x:
        for ov_y in overlaps_y:
            if ov_x.data == ov_y.data and (ov_x.begin <= x1 and
                                           ov_x.end >= x2 and
                                           ov_y.begin <= y1 and
                                           ov_y.end >= y2):
                return True

    return False