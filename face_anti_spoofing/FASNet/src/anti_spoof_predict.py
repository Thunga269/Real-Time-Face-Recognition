import os
import torch
import torch.nn.functional as F


from .model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from .data_io import transform as trans
from .utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}

class AntiSpoofPredict(object):
    def __init__(self, device_id, fasnetv2_path, fasnetv1se_path):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")

        self.fasnetv2_path = fasnetv2_path
        self.fasnetv1se_path = fasnetv1se_path

        self.test_transform = trans.Compose([
            trans.ToTensor(),
        ])

        self.model_fasnetv2 = self._load_model(self.fasnetv2_path)
        self.model_fasnetv1se = self._load_model(self.fasnetv1se_path)

        self.model_fasnetv2.eval()
        self.model_fasnetv1se.eval()

    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        kernel_size = get_kernel(h_input, w_input,)
        model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        return model

    def predict(self, img, model_path):
        img = self.test_transform(img)
        img = img.unsqueeze(0).to(self.device)

        model = model_path.split('_')[-1].split('.pth')[0]
                   
        if model == 'MiniFASNetV2':
            with torch.no_grad():
                result = self.model_fasnetv2.forward(img)
                result = F.softmax(result).cpu().numpy()
        else:
            with torch.no_grad():
                result = self.model_fasnetv1se.forward(img)
                result = F.softmax(result).cpu().numpy()

        return result

