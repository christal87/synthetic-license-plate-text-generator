# Synthetic license plate generator with proper YOLO boxes
# Supports random rotation, perspective, motion blur, photometric augmentations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import random
from tqdm import tqdm
import string

# ---------------- SETTINGS ----------------
OVERALL_ITERATIONS = 40400

LETTERS = string.ascii_uppercase
DIGITS = string.digits
CHARS_MAP = DIGITS + LETTERS + "-"
FONTSIZE_MIN = 36

IMG_WIDTH = 192
IMG_HEIGHT = 64

ROTATION_MIN = -10
ROTATION_MAX = 10

VAL_SPLIT = 0.10

# ---------------- BACKGROUND ----------------
def random_background():
    # choose a basic color background
    color = random.choice([
        (255, 255, 255), # white
        (20,203,242),    # yellowish
        (140,173,139),   # greenish
    ])
    img = np.full((IMG_HEIGHT, IMG_WIDTH, 3), color, dtype=np.uint8)
    return img, color

# ---------------- AUGMENTATIONS ----------------
def apply_dirt(img):
    if random.random() < 0.4:
        noise = np.random.normal(0, 25, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def apply_gaussian_blur(img):
    if random.random() < 0.3:
        k = random.choice([3,5])
        img = cv2.GaussianBlur(img, (k,k), 0)
    return img

def apply_motion_blur(img):
    if random.random() < 0.3:
        size = random.choice([5,7,9])
        kernel = np.zeros((size, size))
        if random.random() < 0.5:
            kernel[int((size-1)/2), :] = np.ones(size)  # horizontal
        else:
            kernel[:, int((size-1)/2)] = np.ones(size)  # vertical
        kernel = kernel / size
        img = cv2.filter2D(img, -1, kernel)
    return img

def apply_lighting_gradient(img):
    if random.random() < 0.3:
        h, w = img.shape[:2]
        gradient = np.tile(
            np.linspace(random.randint(-30,30), random.randint(-30,30), w),
            (h,1)
        )
        for c in range(3):
            img[:,:,c] = np.clip(img[:,:,c].astype(np.int16) + gradient, 0, 255)
        img = img.astype(np.uint8)
    return img

def apply_contrast_variation(img):
    if random.random() < 0.3:
        alpha = random.uniform(0.9,1.1)
        beta = random.randint(-10,10)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

def apply_perspective(img, boxes):
    h, w = img.shape[:2]
    margin = 8
    pts1 = np.float32([[margin,margin],[w-margin,margin],[w-margin,h-margin],[margin,h-margin]])
    shift = 5
    pts2 = pts1 + np.random.randint(-shift, shift+1, pts1.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (w,h), borderValue=(255,255,255))
    # transform bounding boxes
    new_boxes = []
    for x, y, bw, bh in boxes:
        pts = np.array([[x, y],
                        [x+bw, y],
                        [x+bw, y+bh],
                        [x, y+bh]], dtype=np.float32).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        x_min = dst[:,:,0].min()
        y_min = dst[:,:,1].min()
        x_max = dst[:,:,0].max()
        y_max = dst[:,:,1].max()
        new_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
    return img, new_boxes

# ---------------- PLATE GENERATION ----------------
def random_plate():
    if random.random() < 0.5:
        return ''.join(random.choices(LETTERS,k=3)) + "-" + ''.join(random.choices(DIGITS,k=3))
    else:
        return ''.join(random.choices(LETTERS,k=4)) + "-" + ''.join(random.choices(DIGITS,k=3))

def fit_font_to_plate(font_path, plate, max_width, max_height):
    fontsize = max_height
    while fontsize >= FONTSIZE_MIN:
        fnt = ImageFont.truetype(font_path, fontsize)
        ascent, descent = fnt.getmetrics()
        text_height = ascent + descent
        total_width = sum(fnt.getbbox(ch)[2]-fnt.getbbox(ch)[0] + 2 for ch in plate)
        if text_height <= max_height and total_width <= max_width:
            return fnt, total_width, text_height, ascent, descent
        fontsize -= 2
    return fnt, total_width, text_height, ascent, descent

def create():
    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/images/val", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)
    os.makedirs("dataset/labels/val", exist_ok=True)

    class_map = {c:i for i,c in enumerate(CHARS_MAP)}
    fonts = [f for f in os.listdir("fonts") if f.endswith(".ttf")]

    counter = 0
    train_count = 0
    val_count = 0

    pbar = tqdm(total=OVERALL_ITERATIONS*len(fonts), desc="Generating plates")

    for _ in range(OVERALL_ITERATIONS):
        for fontname in fonts:

            plate_text = random_plate()
            font_path = "fonts/" + fontname

            usable_width = IMG_WIDTH - 16
            usable_height = IMG_HEIGHT - 8

            fnt, total_width, text_height, ascent, descent = fit_font_to_plate(font_path, plate_text, usable_width, usable_height)

            bg_img, base_color = random_background()
            img_pil = Image.fromarray(bg_img)
            draw = ImageDraw.Draw(img_pil)

            x_cursor = (IMG_WIDTH - total_width)//2
            baseline = (IMG_HEIGHT - text_height)//2 + ascent

            char_boxes = []
            for ch in plate_text:
                bbox = fnt.getbbox(ch)
                w = bbox[2]-bbox[0]
                h = bbox[3]-bbox[1]
                y_jitter = random.randint(-1,1)
                draw.text((x_cursor, baseline+y_jitter), ch, font=fnt, fill="black", anchor="ls")
                char_boxes.append([x_cursor, baseline - ascent, w, text_height])
                x_cursor += w + 2

            img = np.array(img_pil)

            # Random rotation
            angle = random.uniform(ROTATION_MIN, ROTATION_MAX)
            M = cv2.getRotationMatrix2D((IMG_WIDTH//2, IMG_HEIGHT//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (IMG_WIDTH, IMG_HEIGHT), borderValue=base_color)
            # transform boxes
            for i, (x, y, w_box, h_box) in enumerate(char_boxes):
                pts = np.array([[x, y], [x+w_box, y], [x+w_box, y+h_box], [x, y+h_box]], dtype=np.float32)
                pts = cv2.transform(np.array([pts]), M)[0]
                x_min = pts[:,0].min()
                y_min = pts[:,1].min()
                x_max = pts[:,0].max()
                y_max = pts[:,1].max()
                char_boxes[i] = [x_min, y_min, x_max - x_min, y_max - y_min]

            # Perspective
            img, char_boxes = apply_perspective(img, char_boxes)

            # Photometric augmentations
            img = apply_dirt(img)
            img = apply_gaussian_blur(img)
            img = apply_motion_blur(img)
            img = apply_lighting_gradient(img)
            img = apply_contrast_variation(img)

            # YOLO labels
            yolo_lines = []
            for x, y, w_box, h_box in char_boxes:
                x_min = max(0, x)
                y_min = max(0, y)
                x_max = min(IMG_WIDTH-1, x + w_box)
                y_max = min(IMG_HEIGHT-1, y + h_box)
                xc = (x_min + x_max)/2/IMG_WIDTH
                yc = (y_min + y_max)/2/IMG_HEIGHT
                ww = (x_max - x_min)/IMG_WIDTH
                hh = (y_max - y_min)/IMG_HEIGHT
                yolo_lines.append(f"{class_map[plate_text[char_boxes.index([x, y, w_box, h_box])]]} {xc} {yc} {ww} {hh}")

            # Save
            subset = "val" if random.random()<VAL_SPLIT else "train"
            if subset=="val": val_count+=1
            else: train_count+=1

            img_name = f"plate_{counter}.png"
            label_name = f"plate_{counter}.txt"

            cv2.imwrite(f"dataset/images/{subset}/{img_name}", img)
            with open(f"dataset/labels/{subset}/{label_name}", "w") as f:
                f.write("\n".join(yolo_lines))

            counter += 1
            pbar.update()

    pbar.close()
    print("Train:", train_count, "Val:", val_count, "Total:", counter)


create()
