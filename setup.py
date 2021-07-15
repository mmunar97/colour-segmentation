from setuptools import setup

setup(
    name='colour_segmentation',
    version='1.5',
    packages=['colour_segmentation'],
    url='https://github.com/mmunar97/colour-segmentation',
    license='mit',
    author='marcmunar',
    author_email='marc.munar@uib.es',
    description='Set of algorithms for the color segmentation of an image',
    include_package_data=True,
    install_requires=[
        "scipy",
        "numpy",
        "opencv-python",
        "scikit-image"
    ]
)
