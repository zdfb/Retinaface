cfg_re152 = {
    'name': 'Resnet152',
    'min_sizes': [[16, 20.16, 25.40], [32, 40.32, 50.80], [64, 80.63, 101.59], [128, 161.26, 203.19], [256, 322.54, 406.37]],
    'steps': [4, 8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 16,
    'ngpu': 1,
    'epoch': 150,
    'decay1': 55,
    'decay2': 68,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer1':1, 'layer2': 2, 'layer3': 3, 'layer4': 4},
    'in_channel': 256,
    'out_channel': 256
}

