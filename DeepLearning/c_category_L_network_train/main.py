
from ini_parameters import init_parameters, initialize_velocity
from forward import calc_forward, calc_cost, calc_cost_with_regularization
from backword import calc_backward
from update_w_b import update_parameters_deep
from z_store import z_score_handler
from utils import get_mini_batches, alpha_iteration_by_cos
import  matplotlib.pyplot as plt

def deep_neural_network_model(X, Y, layer_dims, learning_rate=(1e-3, 1e-6), num_iterations=3000, lambda_ = 0):
    """
    终极黑盒：根据你传入的 layer_dims，自动训练任何深度的网络！
    """
    max_rate, min_rate = learning_rate

    # 初始化参数
    parameters = init_parameters(layer_dims)

    # 初始化动量梯度参数
    v, s = initialize_velocity(parameters)

    # 炼丹大循环
    n = 0
    avg_cost_unit = []

    for epoch in range(num_iterations):

        # 进行批量梯度下降， 将整个训练集进行划分
        mini_batches = get_mini_batches(X, Y, batch_size = 64)

        # 在每个 epoch 循环的末尾
        epoch_costs = []


        # 调整学习率
        learning_rate_end = alpha_iteration_by_cos(epoch, num_iterations,  num_iterations // 10, max_rate, min_rate)

        for i, mini_batch in enumerate(mini_batches):
            n = n + 1
            X_batch, Y_batch = mini_batch

            # 向前
            A_last, cache  = calc_forward(X_batch, parameters)
            # 损失
            if lambda_ > 0:
                cost = calc_cost_with_regularization(A_last, Y_batch, parameters, lambda_)
            else:
                cost = calc_cost(A_last, Y_batch)
            # 反向传播
            dcache = calc_backward(Y_batch, parameters, cache, lambda_)


            print(f"Epoch: {epoch} | Batch: {i + 1}/7 | Mini-batch Cost: {cost}")
            epoch_costs.append(cost)

            # 更新参数
            parameters, v, s = update_parameters_deep(parameters, dcache, v, s, n , learning_rate_end)

        avg_cost = sum(epoch_costs) / len(epoch_costs)
        avg_cost_unit.append(avg_cost)
        print(f"Epoch {epoch} 完成 | 该轮平均 Cost: {avg_cost}")


    plt.plot(avg_cost_unit)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning curve")
    plt.show()

    # 返回最终的参数
    return parameters






