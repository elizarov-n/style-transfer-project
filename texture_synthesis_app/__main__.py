import torch
import torchvision
from torchvision import transforms
from PIL import Image

import os

from texture_synthesis import generate_texture

def main():
    print('Укажите путь к примеру текстуры:')
    path = input()
    print()

    sample = transforms.ToTensor()(Image.open(path).convert('RGB'))

    print('Укажите путь для сохранения сгенерированной текстуры:')
    while True:
        save_path = input()

        if not os.path.exists(save_path):
            print('Путь не существует, попробуйте снова')    
            continue
        else:
            break
    print()

    print('Укажите имя сгенерированной текстуры:')
    save_name = input()
    print()
        
    print('Укажите ширину выходной текстуры:')
    width = int(input())
    print()

    print('Укажите высоту выходной текстуры:')
    height = int(input())
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
    result = generate_texture(sample, {1 : 1e6, 6 : 1e6, 11 : 1e6, 20 : 1e6}, width, height, steps, device)

    print('Генерация завершена. Сохранение...')

    torchvision.utils.save_image(result, os.path.join(save_path, save_name))

if __name__ == '__main__':
    main()
