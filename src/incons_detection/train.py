import argparse
import sys
import json
from pathlib import Path
import numpy as np
import warnings

from utils.utils import switch_backend

warnings.filterwarnings("ignore")

def __prepare(loss_type: str, optimizer_type: str, training_instances_path: str, ground_truths_path: str, model_path: str, model_info_path: str):
    # 加载模型和数据
    model = keras.models.load_model(model_path)

    with open(model_info_path, "r") as f:
        model_info = json.load(f)
        input_objects_names = [model_info['model_structure'][str(idx)]['args']['name'] for idx in model_info["input_id_list"]]
        output_layers_names = [model_info['model_structure'][str(idx)]['args']['name'] for idx in model_info["output_id_list"]]

    training_instances = [*np.load(training_instances_path).values()]
    ground_truths = [*np.load(ground_truths_path).values()]

    model.compile(loss=loss_type, optimizer=optimizer_type)  # 使用相同的训练配置

    # feed数据
    # ins = model._feed_inputs + model._feed_targets + model._feed_sample_weights
    # x, y, sample_weight = model._standardize_user_data(training_instances, ground_truths)
    # print(f'2: {x}; {y}; {sample_weight}')

    # model.get_weights() 获取模型的权重，类型 list 保存了每一层的权重
    # model.inputs 输入的 shape 信息
    # model.outputs 输出的 shape 信息
    ins = model.inputs + model.outputs + model.get_weights()

    x, y, sample_weight = keras.utils.unpack_x_y_sample_weight((training_instances, ground_truths))
    if sample_weight is None:
        ins_value = x + y + []

    return model, input_objects_names, output_layers_names, training_instances, ground_truths, ins, ins_value


def __get_outputs(model, input_objects_names, x,
                  output_dir: str, backend: str):
    # 获取所有层的输出
    # get_layer_output = K.function(model._feed_inputs + [K.learning_phase()],
    #                               [layer.output for layer in model.layers if layer.name not in input_objects_names])
    # layers_outputs = get_layer_output(x + [1])

    intermediate_layers = [layer for layer in model.layers if layer.name not in input_objects_names] # 中间层对象
    layers_names   = [layer.name for layer in model.layers if layer.name not in input_objects_names] # 中间层名称

    get_layer_output = Model(inputs=model.input,
                             outputs=[layer.output for layer in intermediate_layers])
    layers_outputs = get_layer_output.predict(x)  # 返回的是所有中间层的输出列表

    def save_outputs(layers_names, layers_outputs, output_dir):
        for name, output in zip(layers_names, layers_outputs):
            save_path = Path(output_dir) / f'{name}.npy'
            np.save(save_path, output)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_outputs(input_objects_names + layers_names, x + layers_outputs, output_dir)

    return {layer_name: output for layer_name, output in zip(input_objects_names + layers_names, x + layers_outputs)}


def __get_loss(model, ins, ins_value,
               loss_path: str, loss_type: str, backend_name: str):

    # # tensorflow
    if backend_name == 'tensorflow':
        y_pred = model(ins_value[0], training=True)
        loss = model.compiled_loss(
            ins_value[1],
            y_pred,
            regularization_losses=model.losses  # 包含 L2 等项
        )

    # pytorch
    if backend_name == 'torch':
        import torch
        torch.nn.L1Loss()
        y_pred = model(ins_value[0], training=True)

        if loss_type == 'mean_squared_error':
            loss_fn = torch.nn.MSELoss()
        elif loss_type == 'mean_absolute_error':
            loss_fn = torch.nn.L1Loss()
        elif loss_type == 'huber_loss':
            loss_fn = torch.nn.SmoothL1Loss()
        elif loss_type == 'categorical_crossentropy':
            loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_type == 'binary_crossentropy':
            loss_fn = torch.nn.BCEWithLogitsLoss()
        elif loss_type == 'kullback_leibler_divergence':
            loss_fn = torch.nn.KLDivLoss()
        elif loss_type == 'poisson':
            loss_fn = torch.nn.PoissonNLLLoss()
        loss = loss_fn(y_pred, torch.tensor(ins_value[1]))


    # 保存loss
    def save_loss(loss, output_path):
        with open(output_path, 'w') as f:
            f.write(str(float(loss)))
    save_loss(loss, loss_path)

    # 获取loss的function
    # get_loss = K.function(
    #     ins + [K.learning_phase()],
    #     [model.total_loss]
    # )
    # loss_value = get_loss(ins_value + [1])[0]


def __get_loss_gradients(model, output_layers_names, ins, ins_value, layers_outputs_value, y,
                         loss_grads_dir: str, model_info_path: str, backend_name: str, loss_type: str):
    # 获取输出层对象，也就是最后一层
    target_layers = [model.get_layer(layer_name) for layer_name in output_layers_names]

    # # 获取d(loss)/d(output)
    # get_loss_grads = K.function(
    #     ins + [K.learning_phase()],
    #     K.gradients(model.total_loss, layer_outputs)
    # )
    # loss_grads_value = get_loss_grads(ins_value + [1])

    loss_grads_value = None

    if backend_name == 'tensorflow':
        import tensorflow as tf

        with tf.GradientTape() as tape:
            outputs = [layer.output for layer in target_layers]
            # 构造一个新的子模型用于获取目标层输出
            submodel = tf.keras.Model(inputs=model.input, outputs=outputs + [model.output])

            # 前向传播，获取目标层的输出以及最终输出
            outputs_values = submodel(ins_value[0], training=True)
            target_layer_outputs = outputs_values[:-1]
            y_pred = outputs_values[-1]

            # 计算损失
            loss = model.compiled_loss(
                ins_value[1],
                y_pred,
                regularization_losses=model.losses  # 包含 L2 等项
            )

        gradients = tape.gradient(loss, target_layer_outputs)
        loss_grads_value = gradients
    elif backend_name == 'torch':
        import torch

        model.train()  # 确保处于训练模式

        input_value = torch.tensor(ins_value[0], dtype=torch.float32, requires_grad=True)
        target_value = torch.tensor(ins_value[1], dtype=torch.float32)

        # 捕获中间层输出和梯度
        layer_outputs = {}
        handles = []

        for layer_name in output_layers_names:
            layer = model.get_layer(layer_name)

            def make_hook(name):
                def hook(module, inp, out):
                    out.retain_grad()
                    layer_outputs[name] = out
                return hook

            h = layer.register_forward_hook(make_hook(layer_name))
            handles.append(h)

        # 前向传播
        preds = model(input_value)
        # 计算 loss
        if loss_type == 'mean_squared_error':
            loss_fn = torch.nn.MSELoss()
        elif loss_type == 'mean_absolute_error':
            loss_fn = torch.nn.L1Loss()
        elif loss_type == 'huber_loss':
            loss_fn = torch.nn.SmoothL1Loss()
        elif loss_type == 'categorical_crossentropy':
            loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_type == 'binary_crossentropy':
            loss_fn = torch.nn.BCEWithLogitsLoss()
        elif loss_type == 'kullback_leibler_divergence':
            loss_fn = torch.nn.KLDivLoss()
        elif loss_type == 'poisson':
            loss_fn = torch.nn.PoissonNLLLoss()
        loss = loss_fn(preds, target_value)
        # 反向传播
        loss.backward()

        # 获取梯度结果
        # grads = {name: out.grad.clone().detach() for name, out in layer_outputs.items()}
        loss_grads_value = [out.grad.clone().detach() for _, out in layer_outputs.items()]
        # 清理 hook
        for h in handles:
            h.remove()


    # 保存d(loss)/d(output)
    def save_loss_grad(layer_names, grads_value, output_dir):
        for layer_name, g in zip(layer_names, grads_value):
            save_path = Path(output_dir) / f'{layer_name}.npy'
            np.save(save_path, g)

    if loss_grads_value is not None:
        # print(f'loss_g: {loss_grads_value}')
        Path(loss_grads_dir).mkdir(parents=True, exist_ok=True)
        save_loss_grad(output_layers_names, loss_grads_value, loss_grads_dir)


if __name__ == "__main__":

    # 正常流程
    # 获取参数
    parse = argparse.ArgumentParser()
    parse.add_argument("--backend", type=str)
    parse.add_argument("--loss", type=str)
    parse.add_argument("--optimizer", type=str)
    parse.add_argument("--model_path", type=str)
    parse.add_argument("--model_info_path", type=str)
    parse.add_argument("--training_instances_path", type=str)
    parse.add_argument("--ground_truths_path", type=str)
    parse.add_argument("--outputs_dir", type=str)
    parse.add_argument("--loss_path", type=str)
    parse.add_argument("--loss_grads_dir", type=str)
    parse.add_argument("--gradients_dir", type=str)
    flags, _ = parse.parse_known_args(sys.argv[1:])

    # # 但文件测试流程
    # from test_flags import MYINFO
    # flags = MYINFO()

    try:
        switch_backend(flags.backend)  # 切换后端
        import keras
        from keras import backend as K
        from keras.models import Model

        FLAG = -1
        model, input_objects_names, output_layers_names, x, y, ins, ins_value = __prepare(flags.loss, flags.optimizer, flags.training_instances_path, flags.ground_truths_path, flags.model_path, flags.model_info_path)

        FLAG = 1
        layers_outputs_value = __get_outputs(model, input_objects_names, x, flags.outputs_dir, flags.backend)
        # layers_outputs_value: 字典，layer_name : outputs

        FLAG = 2 # 获得整体 loss
        __get_loss(model, ins, ins_value, flags.loss_path, flags.loss, flags.backend)

        FLAG = 3
        __get_loss_gradients(model, output_layers_names, ins, ins_value, layers_outputs_value, y, flags.loss_grads_dir, flags.model_info_path, flags.backend, loss_type=flags.loss)

        FLAG = 4
        # __get_gradients(model, input_objects_names, ins, ins_value, layers_outputs_value, y, flags.gradients_dir, flags.model_info_path)
        # FLAG = 5
        # __get_weights(model, x, y, flags.weights_dir)
        # FLAG = 6

        if K.backend() in ['tensorflow', 'torch']:
            K.clear_session()

    except Exception as e:
        import traceback
        print(f'\033[31m异常为: {e} \033[0m')

        # 创建log文件
        log_dir = Path(flags.outputs_dir).parent.parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        with (log_dir / 'detection.log').open(mode='a', encoding='utf-8') as f:
            f.write(f"[ERROR] Crash when training model with {flags.backend}\n")
            traceback.print_exc(file=f)
            f.write("\n\n")

        sys.exit(FLAG)
