# Makefile for image annotation and creating positive samples

# - positive : 	to create and annotate positive images.
# - negative : 	to create and annotate negative images.
# - vec		 : 	to create the vec file.
# - train	 : 	to train the model.
# - clean	 : 	removes all model data, used for retraining model.

# - detect	 : 	run to detect faces in images.
# - reset	 : 	clears the input files for detection.

# Set paths and variables
NEGATIVE_IMAGES_FOLDER 		= training_data/no_faces
POSITIVE_IMAGES_FOLDER		= training_data/faces
NEGATIVE_ANNOTATION_FILE 	= training_data/negative.txt
POSITIVE_ANNOTATION_FILE 	= training_data/positive.txt
POSITIVE_VECTOR_FILE 		= training_data/model.vec

POSITIVE_AMOUNT				= 100
NEGATIVE_AMOUNT				= 120

# Positive
positive:
	python capture.py $(POSITIVE_IMAGES_FOLDER) $(POSITIVE_AMOUNT)
	opencv_annotation \
		--maxWindowHeight=1000 \
		--resizeFactor=3 \
		--annotations=$(POSITIVE_ANNOTATION_FILE) \
		--images=$(POSITIVE_IMAGES_FOLDER)

# Negative
negative:
	python capture.py $(NEGATIVE_IMAGES_FOLDER) $(NEGATIVE_AMOUNT)
	opencv_annotation \
		--maxWindowHeight=1000 \
		--resizeFactor=3 \
		--annotations=$(NEGATIVE_ANNOTATION_FILE) \
		--images=$(NEGATIVE_IMAGES_FOLDER)
	sed -i 's/ 0//' $(NEGATIVE_ANNOTATION_FILE)

# Vec
vec:
	opencv_createsamples \
		-info $(POSITIVE_ANNOTATION_FILE) \
		-bg $(NEGATIVE_ANNOTATION_FILE) \
		-vec $(POSITIVE_VECTOR_FILE) \
		-w 30 \
		-h 30

# Train
train:
	opencv_traincascade \
		-data training_data/xml \
		-vec $(POSITIVE_VECTOR_FILE) \
		-bg $(NEGATIVE_ANNOTATION_FILE) \
		-precalcValBufSize 6000 \
		-precalcIdxBufSize 6000 \
		-numPos $(POSITIVE_AMOUNT) \
		-numNeg $(NEGATIVE_AMOUNT) \
		-w 30 \
		-h 30

# Clear files
clean:
	- rm -f *.txt *.vec *.jpg
	- find training_data/xml -type f -delete
	- find training_data/faces -type f -delete
	- find training_data/no_faces -type f -delete

detect:
	python3 check.py

# Reset
reset:
	- rm -f output/*
	- rm -f input/*
