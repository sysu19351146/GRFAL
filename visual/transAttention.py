import cv2
import importlib
import numpy as np
from PIL import Image
preprocess_image = importlib.import_module(f"src.visualize.utils.image").preprocess_image
show_cam_on_image = importlib.import_module(f"src.visualize.utils.image").show_cam_on_image
cam_method = importlib.import_module(f"src.visualize.transAttention_cam").TransformerCam


def generate(model, opt, prefix='clean'):
    model.eval()
    cam = cam_method(model=model, use_cuda=opt.cuda)

    rgb_img = Image.open(opt.image_path).convert('RGB')
    input_tensor = preprocess_image(rgb_img)
    cam_image = cam(input_tensor=input_tensor, target=None)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(opt.visualize_dir + prefix + "-base_model-" + opt.visual_method + "-" + opt.model_name
                + "-" + opt.visualize_path, cam_image)
    #rgb_img = cv2.imread(opt.image_path, 1)
    #rgb_img = cv2.resize(rgb_img, (224, 224))
    #cv2.imwrite("./examples/origin4.png", rgb_img)