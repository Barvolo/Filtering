from setuptools import setup, find_packages

"""
GEN: ChatGPT.
Prompt: "How to structure setup.py for a Python CLI application with dependencies?"
"""
setup(
    name='edit-image',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    entry_points={
        'console_scripts': [
            'edit-image=edit_image:main',
        ],
    },
    install_requires=[
        'Pillow',
        'numpy',
        'argparse',
    ],
)