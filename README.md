# mirage-decryption

A technical utility for the recovery and generation of steganographic "mirage" images. These files utilize luminance disparities and spatial interlacing to display different content based on preview scaling or background contrast.

## Functionality

* Decryption: Recovers low-luminance hidden data using four distinct image processing tracks.
* Evaluation: Ranks results based on information entropy and standard deviation to identify the most viable output.
* Encryption: Generates mirage files by multiplexing a hidden layer (dark-compressed) and a cover layer (light-expanded).

## Technical Principles

The tool operates on the principle of luminance multiplexing and spatial filtering:

1. Spatial Interlacing: The hidden and cover images are merged using a checkerboard or vertical pixel mask.
2. Luminance Compression: To maintain the "hidden" state under normal viewing, the hidden layer is mapped to the [0, 30] brightness range, while the cover layer is mapped to [155, 255].
3. Recovery Engines:
    * Standard: Linear BGR scaling based on the 98th percentile of pixel intensity.
    * LAB_Adaptive: Local contrast enhancement via CLAHE (Contrast Limited Adaptive Histogram Equalization) within the LAB color space.
    * Vibrant: Value and saturation reinforcement in the HSV space to mitigate gray-out effects.
    * Pure_Gamma: Non-linear inverse gamma transformation ($1/0.45$) for high-depth detail extraction.

## Usage

### Dependencies

* Python 3.7+
* OpenCV
* NumPy

Install requirements:
pip install -r requirements.txt

### Command Line Interface

Decrypting an image:
`python main.py -i input.png -m decrypt`

Processing results are categorized by engine type within the generated _Master_Decrypted directory.

Encrypting an image:
`python main.py -i secret.png -c cover.png -m encrypt -o output.png`

## Implementation Details

* Unicode Support: I/O operations utilize numpy.fromfile to ensure compatibility with non-ASCII file paths.
* Automated Selection: The selection algorithm evaluates the Shannon entropy of the result set to automatically identify the output with the highest information density.