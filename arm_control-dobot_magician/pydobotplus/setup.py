import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pydobotplus',
    packages=['pydobotplus'],
    version='0.1.3',
    description='Python library for Dobot Magician upgraded',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='sammydick22',
    author_email='sjeother@gmail.com',
    url='https://github.com/sammydick22/pydobotplus',
    keywords=['dobot', 'magician', 'robotics', 'm1'],
    classifiers=[],
    install_requires=[
        'pyserial==3.4'
    ]
)
