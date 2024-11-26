import cv2
from pathlib import Path
import argparse
import time
from PIL import Image

from src.lp_recognition import E2E

def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./samples/15.jpg')

    return arg.parse_args()


args = get_arguments()
img_path = Path(args.image_path)
img = cv2.imread(str(img_path))

start = time.time()

model = E2E()

image = model.predict(img)

end = time.time()

print('Model process on %.2f s' % (end - start))

cv2.imshow('License Plate', image)
im = Image.fromarray(image)
im.save("result.jpeg")
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)


cv2.destroyAllWindows()
