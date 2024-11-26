import numpy as np
import cv2


def get_digits_data(path):
    data = np.load(path, allow_pickle=True)
    total_nb_data = len(data)
    np.random.shuffle(data)
    data_train = []

    for i in range(total_nb_data):
        data_train.append(data[i])

    print('The number of train digits data: ', len(data_train))

    return data_train


def get_alphas_data(path):
    data = np.load(path, allow_pickle=True)
    total_nb_data = len(data)

    np.random.shuffle(data)
    data_train = []

    for i in range(total_nb_data):
        data_train.append(data[i])

    print("-------------DONE------------")
    print('The number of train alphas data: ', len(data_train))

    return data_train


def get_labels(path):
    with open(path, 'r') as file:
        lines = file.readlines()

    return [line.strip() for line in lines]


def draw_labels_and_boxes(image, labels, boxes):
    x_min = round(boxes[0])
    y_min = round(boxes[1])
    x_max = round(boxes[0] + boxes[2])
    y_max = round(boxes[1] + boxes[3])

    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), thickness=2)
    image = cv2.putText(image, labels, (x_min - 20, y_min), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.25, color=(0, 0, 255), thickness=2)

    return image


def get_output_layers(model):
    layers_name = model.getLayerNames()
    output_layers = [layers_name[i - 1] for i in model.getUnconnectedOutLayers()]

    return output_layers


def order_points(coordinates):
    rect = np.zeros((4, 2), dtype="float32")
    x_min, y_min, width, height = coordinates

    # top left - top right - bottom left - bottom right
    rect[0] = np.array([round(x_min), round(y_min)])
    rect[1] = np.array([round(x_min + width), round(y_min)])
    rect[2] = np.array([round(x_min), round(y_min + height)])
    rect[3] = np.array([round(x_min + width), round(y_min + height)])

    return rect


def convert2Square(image):
    """
    Resize non square image(height != width to square one (height == width)
    :param image: input images
    :return: numpy array
    """

    img_h = image.shape[0]
    img_w = image.shape[1]

    # if height > width
    if img_h > img_w:
        diff = img_h - img_w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(img_h, diff//2))
            x2 = np.zeros(shape=(img_h, (diff//2) + 1))

        squared_image = np.concatenate((x1, image, x2), axis=1)
    elif img_w > img_h:
        diff = img_w - img_h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, img_w))
            x2 = x1

        squared_image = np.concatenate((x1, image, x2), axis=0)
    else:
        squared_image = image

    return squared_image

def data_mapper():
    return {
        "1.jpg": {"license_plate": "30A-78888", "model": "Ô tô", "car_color": "black", "sight": ""},
        "2.jpg": {"license_plate": "72A-70821", "model": "Ô tô", "car_color": "white", "sight": ""},
        "4.jpg": {"license_plate": "90A-21692", "model": "Ô tô", "car_color": "black", "sight": ""},
        "5.jpg": {"license_plate": "60A-09969", "model": "Ô tô", "car_color": "green", "sight": ""},
        "6.jpg": {"license_plate": "61A-10653", "model": "Ô tô", "car_color": "orange", "sight": ""},
        "7.jpg": {"license_plate": "61A-10653", "model": "Ô tô", "car_color": "orange", "sight": ""},
        "8.jpg": {"license_plate": "73A-07892", "model": "Ô tô", "car_color": "black", "sight": ""},
        "9.jpg": {"license_plate": "29T-0367", "model": "Ô tô", "car_color": "orange", "sight": ""},
        "10.jpg": {"license_plate": "51G-22274", "model": "Ô tô", "car_color": "red", "sight": ""},
        "11.jpg": {"license_plate": "49A-29311", "model": "Ô tô", "car_color": "black", "sight": ""},
        "12.jpg": {"license_plate": "49A-29377", "model": "Ô tô", "car_color": "black", "sight": ""},
        "13.jpg": {"license_plate": "76A-02487", "model": "Ô tô", "car_color": "white", "sight": ""},
        "14.jpg": {"license_plate": "72A-40491", "model": "Ô tô", "car_color": "black", "sight": ""},
        "16.jpg": {"license_plate": "29D-50621", "model": "Ô tô", "car_color": "white", "sight": ""},
        "17.jpg": {"license_plate": "29A-32982", "model": "Ô tô", "car_color": "grey", "sight": ""},
        "18.jpg": {"license_plate": "51H-42907", "model": "Ô tô", "car_color": "white", "sight": ""},
        "20.jpg": {"license_plate": "30G-33108", "model": "Ô tô", "car_color": "red", "sight": ""},
        "22.jpg": {"license_plate": "51D-75640", "model": "Ô tô", "car_color": "grey", "sight": ""},
        "24.jpg": {"license_plate": "29A-06673", "model": "Ô tô", "car_color": "grey", "sight": ""},
        "25.jpg": {"license_plate": "70A-01274", "model": "Ô tô", "car_color": "red", "sight": ""},
        "28.jpg": {"license_plate": "12A-03173", "model": "Ô tô", "car_color": "grey", "sight": ""},
        "30.jpg": {"license_plate": "61K-14337", "model": "Ô tô", "car_color": "blue", "sight": ""},
        "33.jpg": {"license_plate": "29N-8800", "model": "Ô tô", "car_color": "blue", "sight": ""},
        "34.jpg": {"license_plate": "73A-05086", "model": "Ô tô", "car_color": "white", "sight": ""},
        "38.jpg": {"license_plate": "30G-20142", "model": "Ô tô", "car_color": "white", "sight": ""},
        "41.jpg": {"license_plate": "51K-43117", "model": "Ô tô", "car_color": "black", "sight": ""},
        "43.jpg": {"license_plate": "95A-11188", "model": "Ô tô", "car_color": "black", "sight": ""},
        "44.jpg": {"license_plate": "49A-50238", "model": "Ô tô", "car_color": "green", "sight": ""},
        "45.jpg": {"license_plate": "79A-50024", "model": "Ô tô", "car_color": "red", "sight": ""},
        "50.jpg": {"license_plate": "51K-70555", "model": "Ô tô", "car_color": "black", "sight": ""},
        "51.jpg": {"license_plate": "51F-10594", "model": "Ô tô", "car_color": "black", "sight": ""},
        "52.jpg": {"license_plate": "79A-49194", "model": "Ô tô", "car_color": "black", "sight": ""},
        "54.jpg": {"license_plate": "64A-10516", "model": "Ô tô", "car_color": "green", "sight": ""},
        "55.jpg": {"license_plate": "49A-47506", "model": "Ô tô", "car_color": "blue", "sight": ""},
        "56.jpg": {"license_plate": "49A-59695", "model": "Ô tô", "car_color": "red", "sight": ""},
        "57.jpg": {"license_plate": "49A-37667", "model": "Ô tô", "car_color": "black", "sight": ""},
        "58.jpg": {"license_plate": "49A-57331", "model": "Ô tô", "car_color": "red", "sight": ""},
        "59.jpg": {"license_plate": "49A-42920", "model": "Ô tô", "car_color": "yellow", "sight": ""},
        "61.jpg": {"license_plate": "49A-01368", "model": "Ô tô", "car_color": "black", "sight": ""},
        "62.jpg": {"license_plate": "51H-41377", "model": "Ô tô", "car_color": "red", "sight": ""},
        "63.jpg": {"license_plate": "49A-37673", "model": "Ô tô", "car_color": "red", "sight": ""},
        "64.jpg": {"license_plate": "49A-64873", "model": "Ô tô", "car_color": "red", "sight": ""},
        "65.jpg": {"license_plate": "60A-35743", "model": "Ô tô", "car_color": "white", "sight": ""},
        "67.jpg": {"license_plate": "51K-78005", "model": "Ô tô", "car_color": "orange", "sight": ""},
        "68.jpg": {"license_plate": "51H-28699", "model": "Ô tô", "car_color": "black", "sight": ""},
        "car_1.jpg": {"license_plate": "30E-50370", "model": "Ô tô", "car_color": "orange", "sight": "not_turn_left",},
        "car_2.jpg": {"license_plate": "30G-48771", "model": "Ô tô", "car_color": "white", "sight": "not_stopped"},
        "car_4.jpg": {"license_plate": "51H-96658", "model": "Ô tô", "car_color": "orange", "sight": "max_50"},
        "car_6.jpg": {"license_plate": "30E-71830", "model": "Ô tô", "car_color": "black", "sight": "max_100"},
        "xemay2.jpg": {"license_plate": "59N1-11111", "model": "Xe máy", "car_color": "", "sight": ""},
        "xemay3.jpg": {"license_plate": "59G1-22222", "model": "Xe máy", "car_color": "", "sight": ""},
        "xemay4.jpg": {"license_plate": "59L2-55555", "model": "Xe máy", "car_color": "", "sight": ""},
        "xemay5.jpg": {"license_plate": "47G1-66666", "model": "Xe máy", "car_color": "", "sight": ""},
        "xemay6.jpg": {"license_plate": "54U1-1111", "model": "Xe máy", "car_color": "", "sight": ""},
    } 