# Monet Photo Generation using CycleGAN

## Introduction
This project utilizes CycleGAN, a type of Generative Adversarial Network (GAN), to generate Monet-style paintings from real-life photos. CycleGAN is known for its ability to perform style transfer without paired data, making it an excellent choice for tasks like converting photographs into the style of famous artists such as Claude Monet.

## How it Works
CycleGAN consists of two generators and two discriminators. The generators aim to translate images from one domain to another, while the discriminators aim to distinguish between translated images and real images from the target domain. The cycle-consistency loss ensures that the translation is both accurate and visually appealing.

## Requirements
- Python 3
- PyTorch
- NumPy
- OpenCV
- Matplotlib

## Installation
1. Clone the repository: `[git clone https://github.com/yourusername/Monet-Photo-Generation.git](https://github.com/sugam21/GAN-ARS-Div-B-Monet-Photo-Generation-using-Cycle-Gan.git)`
2. Install dependencies: `pip install -r requirements.txt`

## Usage
1. This code  uses google collab to process data and is designed accrodingly.
1. Download your dataset into collab runtime.
2. Create a folder called Academics -> Gan into your main drive.
3. Inside Gan folder create a folder called "saved_checkpoints" to save checkpoints and "saved_images" to store generated images
4. Train the model: Run `python train.py`.

## Customization
- Adjust hyperparameters in `config.py` to fine-tune the training process.
- Experiment with different architectures or loss functions to improve results.

## Contributing
Contributions are welcome! If you have any ideas for improvements or find any issues, feel free to open an issue or create a pull request.

## Credits
This project is inspired by the original CycleGAN paper by Jun-Yan Zhu et al. ([arXiv:1703.10593](https://arxiv.org/abs/1703.10593)).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to the developers of CycleGAN and the open-source community for their contributions.

## Disclaimer
This project is for educational and research purposes only. The generated images may not accurately represent the style of Claude Monet or other artists.
