import setuptools

setuptools.setup(
    name="unet-training",  # Updated name
    version="0.1.0",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "scikit-image==0.23.2",
        "Pillow==10.3.0",
        "tqdm==4.66.4",
        "torch==2.3.0",
        "torchvision==0.18.0",
        "matplotlib==3.8.4",
        "gcsfs==2024.5.0",
        "fsspec==2024.5.0",
    ],
    # You can keep your original author info here
    author="Georgescu Radu Andrei",
    author_email="radugeorgescu2020@gmail.com",
    description="A training application for a multi-head U-Net model on roof segmentation.",
    python_requires=">=3.8",
)