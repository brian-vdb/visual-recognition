# visual-recognition

Assignment for J. D. Mulder

## Annotations

For both face detection and face recognition, we will need to annotate faces within images. To do this, I have created an application that uses an advanced YuNet model to detect faces, after which the person running the tool can decide if all of the faces were annotated correctly. If the annotations were not correct, you can simply reject the annotations and it will skip the image.

To run this program, you simply run:

```bash
python annotation.py --models {models_path} --input {input_path} --output {output_path}
```

- The models path is defined as 'models' if left undefined and it is required to contain a yunet model called 'yunet.onnx'.
- The input path is ignored if left undefined and instead, the application will attempt to start a webcam stream.
- The output path is completely optional and will be 'output' if left undefined.
