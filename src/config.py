from pathlib import Path


class CFG:
    root = Path(__file__).parent.parent.absolute()
    GRAY_THRESHOLD = 150
    CANNY_KERNEL_SIZE = 100
    CHECKBOX_MIN_SIZE = 12
    CHECKBOX_MAX_SIZE = 30
    MIN_OCR_CONFIDENCE = 90
    save_dir = root.joinpath('processed_images')
