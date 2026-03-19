import cv2
import os

# dataset paths
images_dir = "dataset/images/val"
labels_dir = "dataset/labels/val"

# class map (same as generator)
import string
chars_map = string.digits + string.ascii_uppercase + "-"
class_map = {i: c for i, c in enumerate(chars_map)}

files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
files.sort()

index = 0

while True:

    file = files[index]

    img_path = os.path.join(images_dir, file)
    label_path = os.path.join(labels_dir, file.replace(".png", ".txt"))

    img = cv2.imread(img_path)

    if img is None:
        print("Error loading:", file)
        continue

    h_img, w_img = img.shape[:2]

    if os.path.exists(label_path):

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            class_id, xc, yc, w, h = map(float, line.strip().split())

            # YOLO → pixel
            x_center = xc * w_img
            y_center = yc * h_img
            width = w * w_img
            height = h * h_img

            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)

            # draw bbox
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)

            # draw label
            label = class_map[int(class_id)]
            cv2.putText(img, label, (x1, y1-2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 1)

    else:
        print("Missing label:", file)

    # overlay filename + index
    cv2.putText(img, f"{file} ({index+1}/{len(files)})",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255,255,255), 1)

    cv2.imshow("YOLO Plate Viewer", img)

    key = cv2.waitKey(0)

    if key == 27:  # ESC
        break
    elif key == 83 or key == ord('d'):  # right arrow / next
        index = (index + 1) % len(files)
    elif key == 81 or key == ord('a'):  # left arrow / previous
        index = (index - 1) % len(files)

cv2.destroyAllWindows()
