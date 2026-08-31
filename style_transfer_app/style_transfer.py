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

def style_transfer_func(
        content_img : torch.tensor,
        style_img : torch.tensor,
        content_coef : float,
        style_coef : float,
        weights_style : dict,
        layer_reconstruct : int,
        steps : int = 50,
        device : str = 'cpu'
):
    '''
    Функция, генерирующая изображение с контентом content_img в стиле style_img

    Args:
    content_img: изображение, контент которого хотим видеть на сгенерированном
    style_img: изображение, стиль которого хотим перенести на изображение с контентом
    content_coef: значение коэффициента перед лоссом контента
    style_coef: значение коэффициента перед лоссом стиля
    weights_style: значения весов лоссов с разных слоев сверточной сети в итоговом лоссе стиля, 
                если вес не указан для слоя, то он предполагается нулевым
    layer_reconstruct: слой, по выходам которого считается лосс контента
    steps: число шагов генерации
    device: устройство, на котором выполняются вычисления, 'cuda' или 'cpu'

    Returns:
    torch.tensor - результат переноса стиля, на cpu
    '''
    # Загружаем нормализованную VGG-19, оставляем только feature_extractor
    features = torch.load('research_notebooks/normalized_vgg.pt', weights_only=False).features.to(device).eval()    
    
    # Применяем нормализацию изображений для сети VGG
    style_img = normalization(style_img).to(device)
    content_img = normalization(content_img).to(device)

    # Инициализируем шумом итоговое изображение
    x = torch.randn(content_img.shape, requires_grad=True, device=device)

    # Инициализируем оптимизатор нашего изображения
    optimizer = optim.LBFGS([x],
                            lr=1,
                            line_search_fn='strong_wolfe'
                            )

    # Находим, до какого слоя будем прогонять итоговое изображение
    # Будем подразумевать, что если вес указан, то он ненулевой, т.к.
    # по умолчанию неуказанный вес нулевой
    indexes_compute = {i : int((i in weights_style.keys()) and (weights_style[i] != 0)) for i in range(len(features))}
    
    list_for_compute = [i for i in range(len(features)) if indexes_compute[i] == 1]
    count_matrixes = len(list_for_compute)
    last_layer_id = max(list_for_compute)

    flag_process = layer_reconstruct > last_layer_id

    part_model_style = features[:last_layer_id + 1]
    part_model_content = []

    if flag_process:
        part_model_content = features[last_layer_id + 1 : layer_reconstruct + 1]

    # Находим матрицы Грама для изображения стиля
    # и feature map для изображения контента
    source_corrs = [torch.tensor([]) for _ in range(count_matrixes)]
    target_map = torch.tensor([])

    with torch.no_grad():
        find_gram_matrixes(style_img, indexes_compute, source_corrs, part_model_style)
        target_map = features[:layer_reconstruct + 1](content_img)

    torch.cuda.empty_cache()

    x_corrs = [torch.tensor([]) for _ in range(count_matrixes)]

    # Теперь делаем шаги градиента
    for i in range(steps):

        def closure():
                optimizer.zero_grad()

                prev = x.clone()

                corrs_id = 0
                feature_map = torch.tensor([])

                for id in range(last_layer_id + 1):
                        prev = part_model_style[id](prev)

                        if id == layer_reconstruct:
                            feature_map = prev.clone()

                        if indexes_compute[id] == 1:
                            output = prev.clone().flatten(start_dim=-2)
                            x_corrs[corrs_id] = torch.mm(output, output.t()) / output.shape[1]
                            corrs_id += 1

                style_loss = 0
                corrs_id = 0

                for id in range(last_layer_id + 1):
                    if indexes_compute[id] == 1:
                        style_loss += weights_style[id] * ((source_corrs[corrs_id] - x_corrs[corrs_id])**2).mean()
                        corrs_id += 1

                if flag_process:
                    feature_map = part_model_content(prev)

                loss = content_coef * ((feature_map - target_map)**2).mean() + style_coef * style_loss 

                loss.backward()
                return loss

        loss = optimizer.step(closure)

        # Логируем лосс
        print(f'Шаг {i}, loss = {loss}')

    return denormalization(x).detach().cpu()