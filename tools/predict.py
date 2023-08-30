from PIL import Image
from classification import Classification


def image_predict(opt):
    classification = Classification(opt)
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            class_name = classification.detect_image(image)
            print(class_name)

