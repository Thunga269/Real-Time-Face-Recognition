import os.path as osp

this_dir = osp.dirname(__file__)

feature_path = this_dir + '/data/face.bin'
info_path = this_dir + '/data/names.npy'
telegramid_path = this_dir + '/data/telegram_chatids.npy'

model_fasnetv2_path = this_dir + '/weights/FASNet/2.7_80x80_MiniFASNetV2.pth'
model_fasnetv1se_path = this_dir + '/weights/FASNet/4_0_0_80x80_MiniFASNetV1SE.pth'
model_fasnet_path = this_dir + '/weights/FASNet/'

config_fasnet = {
    'model_fasnet_path': model_fasnet_path,
    'model_fasnetv2_path': model_fasnetv2_path,
    'model_fasnetv1se_path': model_fasnetv1se_path
}

port = 8035

config_sendlog = {
    'telegram': {
        'bot': '5660521128:AAHOPhD_wku4ZZiYQVy_aOGJ9Yk2cb1e-_c',
        'chat_id': '-697387438'
    },
    'telegram_warning': {
        'bot': '5809669275:AAHmFZ_fe61VuvKVQmrTRs4d6j9Ie3sGeA0',
        'chat_id': '5809669275'
    },
    'email': {
        'username': 'checkin.systemtesting@gmail.com',
        'password': 'urwigmfadihhoebz'
    },
    'telegram_special': {
        'email_user': '',
        'bot': '',
        'chat_id': ''
    },
    'telegram_bot': '5660521128:AAHOPhD_wku4ZZiYQVy_aOGJ9Yk2cb1e-_c'
}

check_sendmail = True

