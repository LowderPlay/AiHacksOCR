import os
from functools import cmp_to_key

import cv2
import numpy as np
import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader

import imgproc
import test
from craft import CRAFT
from recognition.dataset import AlignCollate, RawDataset
from recognition.model import Model
from recognition.utils import AttnLabelConverter

trained_model_craft = './craft_mlt_25k.pth'
trained_model_recognition = 'recognition/best_accuracy (1).pth'
output = './out/'
cuda = False


def crop(pts, image):
    """
    Takes inputs as 8 points
    and Returns cropped, masked image with a white background
    """
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = image[y:y + h, x:x + w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    return dst2, (y, h, x)


def generate_words(image_name, score_bbox, image):
    num_bboxes = len(score_bbox)
    for num in range(num_bboxes):
        bbox_coords = list(score_bbox.values())[num]
        if bbox_coords.size != 0:
            pts = np.array(bbox_coords).astype('int32')
            pts = np.where(pts < 0, 0, pts)
            if np.all(pts) >= 0:
                word, (y, h, x) = crop(pts, image)

                folder = '/'.join(image_name.split('/')[:-1])

                if not os.path.isdir(os.path.join(output + folder)):
                    os.makedirs(os.path.join(output + folder))

                try:
                    file_name = os.path.join(output + image_name)
                    cv2.imwrite(
                        file_name + '_{}_{}_{}.jpg'.format(y, h, x), word)
                    # print('Image saved to ' + file_name + '_{}_{}.jpg'.format(x, y))
                except:
                    continue


def filter_threshold(data, threshold):
    result = []
    condition = (lambda x, y: abs(x - y) > threshold)
    for element in data:
        if all(condition(element, other) for other in result):
            result.append(element)
    return result


#### MODEL LOADING ####
if __name__ == '__main__':
    net = CRAFT()
    print('Loading weights from checkpoint (' + trained_model_craft + ')')
    if cuda:
        net.load_state_dict(test.copyStateDict(torch.load(trained_model_craft)))
    else:
        net.load_state_dict(test.copyStateDict(torch.load(trained_model_craft, map_location='cpu')))

    net.eval()
    print('model evaluated')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    converter = AttnLabelConverter(
        '0123456789,.?!:&*()%-=+ abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюяABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
    num_class = len(converter.character)
    input_channel = 3
    opt = {
        'Transformation': 'None',
        'Prediction': 'Attn',
        'SequenceModeling': 'BiLSTM',
        'FeatureExtraction': 'ResNet',
        'input_channel': input_channel,
        'output_channel': 512,
        'hidden_size': 256,
        'num_class': num_class,
        'imgH': 32,
        'imgW': 100,
        'batch_size': 192,
        'workers': 4,
        'rgb': True,
        'batch_max_length': 25,
    }
    print('loading recognition model')
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(trained_model_recognition, map_location=device))
    AlignCollate_demo = AlignCollate(imgH=opt['imgH'], imgW=opt['imgW'], keep_ratio_with_pad=False)
#### MODEL LOADING ####

image = imgproc.loadImage(IMAGE)

bboxes, polys, score_text, det_scores = \
    test.test_net(net, image, 0.4, 0.3, 0.4, cuda, False, {'canvas_size': 1280, 'mag_ratio': 1.5})

bbox_score = {}

for box_num in range(len(bboxes)):
    key = str(det_scores[box_num])
    item = bboxes[box_num]
    bbox_score[key] = item

# file_utils.saveResult(image_path, cropped[:, :, ::-1], polys, dirname=output)
# print(bbox_score)
generate_words('test', bbox_score, image)

demo_data = RawDataset(root=output, opt=opt)  # use RawDataset
demo_loader = DataLoader(
    demo_data, batch_size=opt['batch_size'],
    shuffle=False,
    num_workers=int(opt['workers']),
    collate_fn=AlignCollate_demo, pin_memory=True)

model.eval()

results = []
for image_tensors, image_path_list in demo_loader:
    with torch.no_grad():
        batch_size = image_tensors.size(0)
        image_dev = image_tensors.to(device)
        # For max length prediction

        length_for_pred = torch.IntTensor([opt['batch_max_length']] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt['batch_max_length'] + 1).fill_(0).to(device)

        if 'CTC' in opt['Prediction']:
            preds = model(image_dev, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image_dev, text_for_pred, is_train=False)
            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        preds_prob = func.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
    for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
        if 'Attn' in opt['Prediction']:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

        # calculate confidence score (= multiply of pred_max_prob)
        # confidence_score = pred_max_prob.cumprod(dim=0)[-1]
        y_, h_, x_ = img_name[11:-4].split('_')
        # print(f'{x+" "+y:25s}\t {pred:25s}\t {confidence_score:0.4f}')
        results.append([pred, [int(y_), int(h_), int(x_)]])

verticals = np.array(list(map(lambda x: x[1], results)))
threshold = np.average(verticals[:, 1]) * 0.8

rows = np.array(filter_threshold(verticals[:, 0], threshold))
word = [None] * rows.size
for i in results:
    idx = (np.abs(rows - i[1][0])).argmin()
    if word[idx] is None:
        word[idx] = []
    row_ = word[idx]
    row_.append((i[0], i[1][2]))
for i, x in enumerate(word):
    word[i] = ' '.join(map(lambda x: x[0], sorted(x, key=cmp_to_key(lambda item1, item2: item1[1] - item2[1]))))

guess = '\n'.join(word)