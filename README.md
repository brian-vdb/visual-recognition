# visual-recognition

Assignment for J. D. Mulder

## Annotations

For both face detection and face recognition, we will need to annotate faces within images. To do this, I have created an application that uses an advanced YuNet model to detect faces, after which the person running the tool can decide if all of the faces were annotated correctly. If the annotations were not correct, you can simply reject the annotations and it will skip the image.

To run this program, you simply run:

```bash
python src\annotation.py --models {models_path} --input {input_path} --output {output_path}
```

- The models path is defined as 'models' if left undefined and it is required to contain a yunet model called 'yunet.onnx'.
- The input path is ignored if left undefined and instead, the application will attempt to start a webcam stream.
- The output path is completely optional and will be 'output' if left undefined.

## HAAR model training

First you make a vec file using:

```bash
opencv_createsamples -info info.dat -vec trainingfaces_32-32.vec -bg bg.txt -num {num_positives} -h 32 -w 32
```

And then you use this vec file with input images to train the model by calling:

```bash
opencv_traincascade -data cascade -vec trainingfaces_32-32.vec -bg bg.txt -numPos {num_positives} -numNeg {num_negatives} -numStages 32 -acceptanceRatioBreakValue 10e-6 -w 32 -h 32
```

## Main

The most important part of our application is that it needs to be able to use the trained models and parameters to perform face detection and recognition, this can be done in the main using either an input folder or a webcam stream. This all depends on the wishes for how the user wants to test their models and it can be switched based on if you provide an input folder or not. During the execution, recognized faces will be shown in green with the label assosicated with the face, while unrecognized detections are drawn with red. These are typically False Positives, but due to the nature of Eigenfaces, normal faces are sometimes also hard to recognize due to rotations or ambient lightning differences.

To tun this program, you simply run:

```bash
python src\main.py --models {model_path} --input {input_path}
```

- The models path is defined as 'models' if left undefined and it is required to contain a the following files: 'cascade.xml', 'target_shape.npy', 'mean_face.npy', 'best_eigenfaces.npy' and 'recognizer.yml'. Cascade.xml can either be found online or you can train one yourself. The other files are used for eigenfaces and can be created using the 'face_recognition.ipynb' notebook.
- The input path is ignored if left undefined and instead, the application will attempt to start a webcam stream.
