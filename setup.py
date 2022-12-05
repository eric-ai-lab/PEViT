import setuptools

VERSION = '0.0.1'

setuptools.setup(name='vision_benchmark',
                 author='chunyl',
                 author_email='chunyl@microsoft.com',
                 version=VERSION,
                 python_requires='>=3.6',
                 packages=setuptools.find_packages(exclude=['test', 'test.*']),
                 package_data={'': ['resources/*']},
                 install_requires=[
                     'yacs~=0.1.8',
                     'scikit-learn',
                     'timm>=0.3.4',
                     'numpy>=1.18.0',
                     'sharedmem',
                     'torch>=1.7.0',
                     'PyYAML~=5.4.1',
                     'Pillow',
                     'torchvision>=0.8.0',
                     'vision-datasets>=0.2.0',
                     'vision-evaluation>=0.2.2',
                     'tqdm~=4.62.3',
                     'transformers~=4.11.3'
                 ],
                 entry_points={
                     'console_scripts': [
                         'vb_linear_probe=vision_benchmark.commands.linear_probe:main',
                         'vb_zero_shot_eval=vision_benchmark.commands.zeroshot_eval:main',
                         'vb_eval=vision_benchmark.commands.eval:main',
                         'vb_submit_to_leaderboard=vision_benchmark.commands.submit_predictions:main',
                         'vb_image_caption_eval=vision_benchmark.commands.image_caption_eval:main',
                     ]
                 })
