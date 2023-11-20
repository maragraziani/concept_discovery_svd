from setuptools import setup

setup(
    name='cdisco',
    version='0.1.0',    
    description='Concept discovery with Singular Value Decomposition',
    url='https://github.com/maragraziani/cdisco',
    author='Mara Graziani',
    author_email='mara.graziani@hevs.ch',
    license='MIT',
    packages=['scripts'],
    install_requires=['numpy', 'matplotlib',                    
                      'scikit-learn==1.2.2','torch==2.0.1',
'torchvision==0.15.2'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
