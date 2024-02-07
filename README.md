# Real-time-Detection-of-Motorcyclists-Without-Helmets

The primary goal of this project is to implement real-time detection of motorcyclists without helmets. While there have been similar initiatives in this field, most suffer from sluggish performance, rendering them unsuitable for real-time applications.

To address this challenge, I've devised an alternative pipeline leveraging threading to enable real-time detection and the results are very promising.

The approach involves concurrently running two YOLOv5 object detection models: one for motorcycle detection (pre-trained YOLOv5s model) and another for head/helmet detection (custom-trained YOLOv5m model). The outputs of these two models are subsequently passed to a function which detects motorcyclists without helmets.

The custom training of YOLOv5m utilized a dataset comprising 11,000 images, meticulously prepared with two classes: head and helmet. This dataset was prepared in RoboFlow. Notably, the model achieved an impressive mAP50 accuracy of 96.9%.

Initially, I considered using the multiprocessing module, but faced some issues, leading me to use threading instead.

In this version, I haven't implemented motorcyclist tracking, but will do so in the next version.
