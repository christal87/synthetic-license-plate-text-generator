# Synthetic license plate dataset generator

Creates a YOLO compatible dataset of random license plate text strings consisting of uppercase latin alphabet characters and a dash (37 different classes in total) using standardized TrueType fonts for further experimentation. New Hungarian license plates (introduced July 2022) use a specific, modernized, forgery-resistant, sans-serif font designed to comply with EU standards. While earlier plates often used variants of [DIN 1451](https://en.wikipedia.org/wiki/DIN_1451), the new 4-letter/3-digit format utilizes a unique, font that shares characteristics with [FE-Schrift](https://en.wikipedia.org/wiki/FE-Schrift), but is customized for Hungarian regulations.  

Output images are sized 192x64 pixels containing ABCD-123, ABC-123 formatted text rendered on a white, yellow or green canvas to simulate plate background colors. Before saving they receive multiple crude photometric augmentations:  
- dirt-like noise
- Gaussian blur
- motion blur
- lighting gradient fluctuation
- contrast variation
- perspective transformation

YOLO compatible bounding box classes and coordinates are added beforehand taken by PIL ImageFont's built-in getbbox() method and recomputed after augmentations when needed. There's an additional simple opencv based viewer/browser for taking visual inspections before training.  

The dataset gets split 90%-10% into training and validation sets.  

This is how synthetic images look:  
![Generator examples](/examples/synth_plates.png "Generator examples")

Training metrics for 26n model after 35 iterations (patience 7) on 80800 generated images and YOLO built-in augmentations disabled(classification loss seems to become quickly overfit vs box loss):  
![Training metrics](/examples/metrics_35it.png "Training metrics")

Some inferece results:  
![Inference examples](/examples/real_plates.png "Inference examples")
