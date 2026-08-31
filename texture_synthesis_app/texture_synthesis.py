import torch
from torch import optim

from process_image import normalization, denormalization 

def find_gram_matrixes(input_image : torch.tensor,
                       indexes_compute : dict,
                       result_list : list,
                       model):
    '''
    Функция для вычисления матриц Грама по выходам сверточных слоев

    Args:
    input_image: входное изображение для сети
    indexes_compute: словарь, содержащий информацию о тех слоях,
        по выходам которых нужно считать матрицы Грама;
        значение 0, если не нужно;
        значение 1, если нужно;
    result_list: итоговый список, куда нужно поместить результаты вычислений
    model: модель, по слоям которой проходит входное изображение

    Returns: void
    '''
    prev = input_image.clone()
    output = torch.tensor([])

    index_res = 0
    for id in range(len(model)):
        prev = model[id](prev)

        if indexes_compute[id] == 1:
            output = prev.flatten(start_dim = -2)
            result_list[index_res] = torch.mm(output, output.t()) / output.shape[1]
            index_res += 1
     

def generate_texture(
        input_img : torch.tensor,
        weights_losses : dict,
        width : int,
        height : int,
        steps : int = 50,
        device : str = 'cpu'
):
    '''
    Функция, генерирующая текстуру произвольного размера по примеру

    Args:
    input_img: входной пример текстуры для генерации, RGB
    weight_losses: словарь учитываемых лоссов в итоговом, если вес не указан для слоя,
                то он предполагается нулевым
    width: ширина выходного изображения
    height: высоты выходного изображения
    steps: число шагов оптимизации в генерации текстуры
    device: устройство для вычислений, 'cpu' или 'cuda'

    Returns:
    torch.tensor - сгенерированная текстуру, на cpu
    '''
    
    # Загружаем нормализованную VGG-19, оставляем только feature_extractor
    features = torch.load('research_notebooks/normalized_vgg.pt', weights_only=False).features.to(device).eval()
    
    # Применяем нормализацию для сети VGG
    source_img = normalization(input_img).to(device)

    # Инициализируем шумом итоговое изображение
    x = torch.randn((input_img.shape[0], height, width), requires_grad=True, device=device)

    # Инициализируем оптимизатор нашего изображения
    optimizer = optim.LBFGS([x],
                            lr=1,
                            line_search_fn='strong_wolfe'
                            )

    # Находим, до какого слоя будем прогонять итоговое изображение
    # Будем подразумевать, что по умолчанию неуказанный вес нулевой
    indexes_compute = {i : int((i in weights_losses.keys()) and (weights_losses[i] != 0)) for i in range(len(features))}

    list_for_compute = [i for i in range(len(features)) if indexes_compute[i] == 1]
    count_matrixes = len(list_for_compute)
    last_layer_id = max(list_for_compute)

    part_model = features[:last_layer_id + 1]

    # Находим матрицы Грама для исходной текстуры
    source_corrs = [torch.tensor([]) for _ in range(count_matrixes)]

    with torch.no_grad():
        find_gram_matrixes(source_img, indexes_compute, source_corrs, part_model)

    torch.cuda.empty_cache()

    x_corrs = [torch.tensor([]) for _ in range(count_matrixes)]

    # Теперь делаем шаги градиента
    for i in range(steps):

        def closure():
                optimizer.zero_grad()

                find_gram_matrixes(x, indexes_compute, x_corrs, part_model)

                loss = 0
                corrs_id = 0

                for id in range(last_layer_id + 1):
                    if indexes_compute[id] == 1:
                        loss += weights_losses[id] * ((source_corrs[corrs_id] - x_corrs[corrs_id])**2).mean()
                        corrs_id += 1

                loss.backward()
                return loss

        loss = optimizer.step(closure)

        # Логируем лосс
        print(f'Шаг {i}, loss = {loss}')

    return denormalization(x).detach().cpu()