import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='qsa',
    version='1.0',
    author='Christophe Magnani',
    description='Linear and Quadratic Analysis in the Frequency Domain',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://openqsa.org',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'matplotlib'],
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Development Status :: 1 - Planning'
    ]
)
