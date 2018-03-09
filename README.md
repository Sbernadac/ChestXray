# ChestXray
Machine-learning on XRay

Tree directories :

Project/images/ 		contains all original images

Project/imagesAugmented/	contains all augmented images

Project/Data/			contains all scripts to split data set between Test and Training/Validation

Project/Simple/onePathology	conyains all scripts to manage simple pathology detection

Project/Simple/transferLearning contains all scripts to manage transfer learning

Project/multiPathologies	contains all scripts to manage multi pathologies detection


Python scripts parameters:

-p <pathology name>
  
-t <test type: between 'test' (all test dataset 11145) 'random' (2000 extracted from test dataset) 'image name' (for simple test)>

-s <shape value: image height and weight size>
  
-d <dimension value: 1 for grayscale 3 for rgb>

-m <model name (for predict scripts)>



Usage example:
python AlexNetLike.py -p Cardiomegaly -s 2000
python transferPredict.py  -p Cardiomegaly -t test  -s 224 -d 3 -m myCardiomegaly10000.h5
