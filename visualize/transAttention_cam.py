import torch
import numpy as np
from .utils.image import show_cam_on_image

class TransformerCam:
    def __init__(self,
                 model: torch.nn.Module,
                 use_cuda: bool = False) -> None:
        self.model = model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

    #rule 5 from paper
    def avg_heads(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        return cam

    # rule 6 from paper
    def apply_self_attention_rules(self, R_ss, cam_ss):
        R_ss_addition = torch.matmul(cam_ss, R_ss)
        return R_ss_addition

    def forward(self,
                input: torch.Tensor,
                class_index):
        if self.cuda:
            input = input.cuda()
        output = self.model(input, register_hook=True)
        if class_index is None:
            class_index = np.argmax(output.cpu().data.numpy(), axis=-1)
        print("predicted_class:", class_index)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, class_index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        one_hot = torch.sum(one_hot * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        cam_image = self.generate_visualization(input)
        return cam_image


    def generate_visualization(self, original_image):
        iH, iW = original_image.shape[-2], original_image.shape[-1]
        transformer_attribution = self.generate_relevance()
        mH, mW = int(transformer_attribution.shape[0]**0.5), int(transformer_attribution.shape[0]**0.5)
        transformer_attribution = transformer_attribution.reshape(1, 1, mH, mW)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=iW//mW,
                                                                  mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(iH, iW).data.cpu().numpy()
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
        #transformer_attribution[transformer_attribution<0.1] = transformer_attribution[transformer_attribution<0.1] * 2
        image_transformer_attribution = original_image.squeeze().permute(1, 2, 0).data.cpu().numpy()
        image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())

        cam_image = show_cam_on_image(image_transformer_attribution,transformer_attribution, use_rgb=True)

        return cam_image


    def generate_relevance(self):
        num_tokens = self.model.transformer.encoder_layers[0].attn.get_attention_map().shape[-1]
        R = torch.eye(num_tokens, num_tokens)
        if self.cuda:
            R = R.cuda()
        for blk in self.model.transformer.encoder_layers:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attention_map()
            cam = self.avg_heads(cam, grad)
            R += self.apply_self_attention_rules(R, cam)
        return R[0, 1:]

    def __call__(self, input_tensor: torch.Tensor,target = None):
        return self.forward(input_tensor,target)