# =====[ IMPORTS ]=====
import os
import cv2
import torch
import numpy as np

from torchvision import transforms
from emotic import Emotic


# =====[ PROCESS IMAGES ]=====
def process_images(context_norm, body_norm, image_context_path=None, image_context=None, image_body=None, bbox=None):

    # Error handling
    if image_context is None and image_context_path is None:
        raise ValueError('Both image_context and image_context_path cannot be none. Please specify one of the two.')

    if image_body is None and bbox is None:
        raise ValueError('Both body image and bounding box cannot be none. Please specify one of the two')

    # Load image
    if image_context_path is not None:
        image_context = cv2.cvtColor(cv2.imread(image_context_path), cv2.COLOR_BGR2RGB)

    # Apply bounding box
    if bbox is not None:
        image_body = image_context[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()

        image_context = cv2.resize(image_context, (224, 224))
        image_body = cv2.resize(image_body, (128, 128))

        test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        body_norm = transforms.Normalize(body_norm[0], body_norm[1])

        image_context = context_norm(test_transform(image_context)).unsqueeze(0)
        image_body = body_norm(test_transform(image_body)).unsqueeze(0)

        return image_context, image_body


# =====[ INFER ]=====
def infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models, image_context_path=None, image_context=None, image_body=None, bbox=None, to_print=True):

    # Loading and processing images and models
    image_context, image_body = process_images(context_norm, body_norm, image_context_path=image_context_path, image_context=image_context, image_body=image_body, bbox=bbox)
    model_context, model_body, emotic_model = models

    # Performing inference
    with torch.no_grad():
        image_context = image_context.to(device)
        image_body = image_body.to(device)

        pred_context = model_context(image_context)
        pred_body = model_body(image_body)

        pred_cat, pred_cont = emotic_model(pred_context, pred_body)
        pred_cat = pred_cat.squeeze(0)
        pred_cont = pred_cont.squeeze(0).to('cpu').data.numpy()

        bool_cat_pred = torch.gt(pred_cat, thresholds)

        # List emotions
        emotions = [ind2cat[i] for i in range(len(bool_cat_pred)) if bool_cat_pred[i]]

        return emotions, list(pred_cont * 10)


# =====[ PREDICT ]=====
def predict(image_path, x1, y1, x2, y2):

    # Setup
    configuration_path = '../configuration'
    thresholds_path = os.path.join(configuration_path, 'thresholds')
    model_path = os.path.join(configuration_path, 'models')

    context_mean = [0.4690646, 0.4407227, 0.40508908]
    context_std = [0.2514227, 0.24312855, 0.24266963]
    body_mean = [0.43832874, 0.3964344, 0.3706214]
    body_std = [0.24784276, 0.23621225, 0.2323653]
    context_norm = [context_mean, context_std]
    body_norm = [body_mean, body_std]

    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness',
           'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    vad = ['Valence', 'Arousal', 'Dominance']
    cat2ind, ind2cat, ind2vad = {}, {}, {}

    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion

    for idx, continuous in enumerate(vad):
        ind2vad[idx] = continuous

    gpu = 0

    # Choose GPU if available, CPU otherwise
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    thresholds = torch.FloatTensor(np.load(os.path.join(thresholds_path, 'val_thresholds.npy'))).to(device)
    model_context = torch.load(os.path.join(model_path, 'model_context1.pth')).to(device)
    model_body = torch.load(os.path.join(model_path, 'model_body1.pth')).to(device)

    # Initiate models
    emotic_model = torch.load(os.path.join(model_path, 'model_emotic1.pth')).to(device)
    model_context.eval()
    model_body.eval()
    emotic_model.eval()
    models = [model_context, model_body, emotic_model]

    # Running inference
    bbox = [x1, y1, x2, y2]
    categories, continuous = infer(context_norm, body_norm, ind2cat, ind2vad, device, thresholds, models, image_context_path=image_path, bbox=bbox)

    # Return results
    results = {'emotions': categories, 'valence': continuous[0], 'arousal': continuous[1], 'dominance': continuous[2]}

    return results


# =====[ TESTING ]=====
result = predict('../data/1633203144.716117.jpg', 537, 322, 823, 608)
print(result)
