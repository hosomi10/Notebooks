# Notebooks
Jetson Nano
Collision_avoidance  
  
Create 'data' folder for pictures of 224 x 224 pixels and 300 x 300 pixels.  
Upload alexnet, mobilenet, and tflite runtime classification folders in 'test' folder  
  
Tasks(6/13)
===
* Check memory pressure while executing alexnet,mobilenet,tflite on Jetbot.  
* Decide capture image size(300x300 or 300x400)
* Mapping (path search) by using another camera?
* Determine an object is near/far, from an image

classification results(6/13)
===

banana.jpg(224x224 and 300x300)
---
* mobilenet(model_test.py)  
954: 'banana', 				31.62  
502: 'clog, geta, patten, sabot', 19.17  
666: 'mortar', 				10.36  
659: 'mixing bowl', 			6.30  
464: 'buckle', 				4.60

* alexnet(alexnet_model_test.py)  
954: 'banana', 			59.43  
666: 'mortar', 			9.54  
943: 'cucumber, cuke', 	2.18  
939: 'zucchini, courgette', 	2.08  
941: 'acorn squash', 		2.02  

* tflite-runtime(tflite_detect_loop_cv.py)  
Class=banana  
Probability=0.5625  
Location=(83,107)-(217,186)  
Class=chair  
Probability=0.5234375  
Location=(54,5)-(267,199)  

orange1.jpg
---
* mobilenet  
722: 'ping-pong ball', 	66.45  
950: 'orange', 		24.56  
522: 'croquet ball', 	4.55  
674: 'mousetrap', 	0.58  
951: 'lemon', 			0.46

* alexnet  
722: 'ping-pong ball', 	86.51  
950: 'orange', 		1.78  
522: 'croquet ball', 	1.32  
941: 'acorn squash', 	1.20  
720: 'pill bottle', 		1.10  

* tflite-runtime  
Class=orange  
Probability=0.79296875  
Location=(97,117)-(188,238)  
Class=chair  
Probability=0.57421875  
Location=(37,23)-(264,226)  
Class=dining table  
Probability=0.55078125
Location=(-6,138)-(297,299)  
