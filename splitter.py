import os, shutil, sys

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = './'

if not os.path.exists(path + 'ch2_ed'):
    os.mkdir(path + 'ch2_ed')
if not os.path.exists(path + 'ch2_es'):
    os.mkdir(path + 'ch2_es')
if not os.path.exists(path + 'ch4_ed'):
    os.mkdir(path + 'ch4_ed')
if not os.path.exists(path + 'ch4_es'):
    os.mkdir(path + 'ch4_es')

images = sorted(os.listdir(path))

for ii, image in enumerate(images):
    if not image.endswith('.png'):
        continue
    if ('ch2' in image) and ('_ed_' in image):
        shutil.copy(path + image, path + 'ch2_ed/' + image)
    elif ('ch2' in image) and ('_es_' in image):
        shutil.copy(path + image, path + 'ch2_es/' + image)
    elif ('ch4' in image) and ('_ed_' in image):
        shutil.copy(path + image, path + 'ch4_ed/' + image)
    elif ('ch4' in image) and ('_es_' in image):
        shutil.copy(path + image, path + 'ch4_es/' + image)
