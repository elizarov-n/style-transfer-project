import torch
import torchvision
from torchvision import transforms
from PIL import Image

import os

from style_transfer import style_transfer_func

def main():
    print('Укажите путь к изображению контента:')
    content_path = input()
    print()

    content_img = transforms.ToTensor()(Image.open(content_path).convert('RGB'))

    print('Укажите путь к изображению стиля:')
    style_path = input()
    print()

    style_img = transforms.ToTensor()(Image.open(style_path).convert('RGB'))

    ratio = style_img.shape[1] / style_img.shape[2]
    new_size = min(content_img.shape[1:])
    style_img = transforms.Resize(min(int(ratio * new_size), int((1 / ratio) * new_size)))(style_img)

    print('Укажите путь для сохранения сгенерированного изображения:')
    while True:
        save_path = input()

        if not os.path.exists(save_path):
            print('Путь не существует, попробуйте снова')    
            continue
        else:
            break
    print()

    print('Укажите имя сгенерированного изображения:')
    save_name = input()
    print()

    print('Укажите режим: [content, balance, style]\nПояснение:\ncontent - сдвиг генерации в сторону контента\nbalance - баланс между стилем и контентом\nstyle - сдвиг генерации в сторону стиля')
    mode = input()

    layer_reconstruct = 20

    if mode == 'content':
        layer_reconstruct = 11
    if mode == 'style':
        layer_reconstruct = 29
    print()

    print('Укажите число шагов генерации:\nПрим. Чем больше шагов, тем качественней выйдет текстура, однако время может кратно возрасти')
    steps = int(input())
    print()

    device = 'cpu'
    if torch.cuda.is_available():
        print('На вашем устройстве доступны вычисления с помощью видеокарты')
        print('Выберите устройство для вычислений: [cpu, cuda]')
        device = input()
    print()

    print('Генерация начинается...')
    result = style_transfer_func(content_img,
                                 style_img,
                                 1e5, 1e6,
                                 {1 : 1, 6 : 1, 11 : 1, 20 : 1},
                                 layer_reconstruct,
                                 steps,
                                 device)

    print('Генерация завершена. Сохранение...')

    torchvision.utils.save_image(result, os.path.join(save_path, save_name))

if __name__ == '__main__':
    main()
