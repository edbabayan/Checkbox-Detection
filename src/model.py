from collections import defaultdict

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from tqdm import tqdm

from src.config import CFG
from src.utils import (build_interval_trees, is_surrounded,
                       checkbox_in_ocr_region, most_frequent,
                       pdf_to_image)


class CheckboxDetector:
    def __init__(self, config):
        self.config = config

    def detect_checkboxes(self, pdf_path):
        pages = pdf_to_image(pdf_path)
        results = defaultdict(list)
        all_heights = []

        for page_num, page_image in tqdm(list(enumerate(pages, start=1)), desc="Processing pages"):
            ocr_boxes = self._get_ocr_boxes(page_image)
            if not ocr_boxes:
                continue

            all_heights.extend(self._get_box_heights(ocr_boxes))
            height = most_frequent(all_heights)
            max_checkbox, min_checkbox = height * 1.5, height * 0.5
            ocr_tree_x, ocr_tree_y = build_interval_trees(ocr_boxes)

            img_bin_final = self._get_binary_image(page_image)
            detections, checkboxes = self._detect_checkboxes_in_image(img_bin_final, page_image, ocr_tree_x, ocr_tree_y,
                                                                      ocr_boxes, max_checkbox, min_checkbox)

            results[page_num].extend(checkboxes)

        return results

    def _get_ocr_boxes(self, page_image):
        ocr = pytesseract.image_to_data(page_image, output_type=Output.DICT)
        ocr_boxes = []

        for i in range(len(ocr["text"])):
            x = ocr["left"][i]
            y = ocr["top"][i]
            w = ocr["width"][i]
            h = ocr["height"][i]
            text = ocr["text"][i]
            conf = int(ocr["conf"][i])

            if conf > self.config.MIN_OCR_CONFIDENCE:
                text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                if len(text) > 0:
                    ocr_boxes.append((x, y, x + w, y + h))

        return ocr_boxes

    @staticmethod
    def _get_box_heights(ocr_boxes):
        return [y2 - y1 for _, y1, _, y2 in ocr_boxes]

    def _get_binary_image(self, page_image):
        gray_scale = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
        _, img_bin = cv2.threshold(gray_scale, self.config.GRAY_THRESHOLD, 255, cv2.THRESH_BINARY)
        img_bin = ~img_bin

        kernel_pre = np.ones((1, 1), np.uint8)
        img_bin = cv2.erode(img_bin, kernel_pre, iterations=1)

        line_min_width = self.config.CHECKBOX_MIN_SIZE
        kernel_h = np.ones((1, line_min_width), np.uint8)
        kernel_v = np.ones((line_min_width, 1), np.uint8)

        img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_h)
        img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_v)

        img_bin_final = img_bin_h | img_bin_v

        kernel_post = np.ones((3, 3), np.uint8)
        img_bin_final = cv2.dilate(img_bin_final, kernel_post, iterations=1)

        return img_bin_final

    def _detect_checkboxes_in_image(self, img_bin_final, page_image, ocr_tree_x, ocr_tree_y, ocr_boxes, max_checkbox,
                                    min_checkbox):
        gray_scale = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
        detections = 0
        checkboxes = []

        for x, y, w, h, area in stats[2:]:
            if self._is_valid_checkbox(w, h) and not is_surrounded(ocr_tree_x, ocr_tree_y, (x, y, x + w, y + h)):
                checkbox = self._process_checkbox(x, y, w, h, gray_scale, page_image, ocr_boxes)
                if checkbox:
                    checkboxes.append(checkbox)
                    detections += 1

        if detections == 0:
            checkboxes = self._detect_fallback_checkboxes(stats, gray_scale, ocr_tree_x, ocr_tree_y, ocr_boxes,
                                                          max_checkbox, min_checkbox)

        return detections, checkboxes

    def _is_valid_checkbox(self, w, h):
        return self.config.CHECKBOX_MIN_SIZE <= w <= self.config.CHECKBOX_MAX_SIZE and self.config.CHECKBOX_MIN_SIZE <= h <= self.config.CHECKBOX_MAX_SIZE

    @staticmethod
    def _process_checkbox(x, y, w, h, gray_scale, ocr_boxes):
        x1, y1, x2, y2 = x, y, x + w, y + h
        checkbox = x1, y1, x2, y2

        if not checkbox_in_ocr_region(checkbox, ocr_boxes):
            sliced_image = gray_scale[y1:y2, x1:x2]
            sliced_image[sliced_image == 0] = 1
            sliced_image[sliced_image == 255] = 0
            height, width = sliced_image.shape
            pixels = height * width

            if 0.05 <= sliced_image.sum() / pixels <= 0.75:
                return {'class': 'Marked', 'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}}
            elif sliced_image.sum() / pixels <= 0.75:
                return {'class': 'Unmarked', 'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}}

        return None

    def _detect_fallback_checkboxes(self, stats, gray_scale, ocr_tree_x, ocr_tree_y, ocr_boxes, max_checkbox,
                                    min_checkbox):
        checkboxes = []

        for x, y, w, h, area in stats[2:]:
            if (self._is_valid_fallback_checkbox(w, h, max_checkbox, min_checkbox)
                    and not is_surrounded(ocr_tree_x,
                                          ocr_tree_y, (
                                                  x, y,
                                                  x + w,
                                                  y + h))):
                x1, y1, x2, y2 = x, y, x + w, y + h
                checkbox = x1, y1, x2, y2

                if not checkbox_in_ocr_region(checkbox, ocr_boxes):
                    sliced_image = gray_scale[y1:y2, x1:x2]
                    sliced_image[sliced_image == 0] = 1
                    sliced_image[sliced_image == 255] = 0
                    height, width = sliced_image.shape
                    pixels = height * width

                    if 0.05 <= sliced_image.sum() / pixels <= 0.75:
                        checkboxes.append({'class': 'Marked', 'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}})
                    elif sliced_image.sum() / pixels <= 0.75:
                        checkboxes.append(
                            {'class': 'Unmarked', 'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}})

        return checkboxes

    @staticmethod
    def _is_valid_fallback_checkbox(w, h, max_checkbox, min_checkbox):
        return min_checkbox <= w <= max_checkbox and min_checkbox <= h <= max_checkbox


_pdf_path = "/home/eduard/Downloads/CAZ PEO D5 - Final Sub Doc Form.pdf"
detector = CheckboxDetector(CFG)
checkboxes = detector.detect_checkboxes(_pdf_path)
print(checkboxes)
