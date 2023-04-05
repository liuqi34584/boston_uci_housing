import paddle.fluid as fluid
import paddle
import numpy as np
import os
import matplotlib.pyplot as plt

# 网页端paddle注释本行
paddle.enable_static()


BUF_SIZE=500
BATCH_SIZE=512

# 每次从缓存中随机读取批次大小的数据
train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=BUF_SIZE),batch_size=BATCH_SIZE)
test_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.test(),buf_size=BUF_SIZE),batch_size=BATCH_SIZE)

# x表示13维的特征值，y表示目标值
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')

#定义一个简单的线性网络,连接输入和输出的全连接层
#input:输入tensor;size:该层输出单元的数目;act:激活函数
y_predict=fluid.layers.fc(input=x,size=1,act=None)

cost = fluid.layers.square_error_cost(input=y_predict, label=y) #求一个batch的损失值
avg_cost = fluid.layers.mean(cost)  #对损失值求平均值
optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.78)
opts = optimizer.minimize(avg_cost)
test_program = fluid.default_main_program().clone(for_test=True)


use_cuda = False    
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)              #创建一个Executor实例exe
exe.run(fluid.default_startup_program()) #Executor的run()方法执行startup_program(),进行参数初始化

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])  # feed_list:向模型输入的变量表或变量表名


iter=0
iters=[]
train_costs=[]
EPOCH_NUM=25
model_save_dir = "./fit_a_line.inference.model"

for pass_id in range(EPOCH_NUM):     
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):       
        # 喂入一个batch的训练数据，根据feed_list和data提供的信息，将输入数据转成一种特殊的数据结构      
        train_cost = exe.run(program=fluid.default_main_program(), feed=feeder.feed(data), fetch_list=[avg_cost])  
        iter=iter+BATCH_SIZE
        iters.append(iter)
        train_costs.append(train_cost[0][0])

    test_cost = 0
    for batch_id, data in enumerate(test_reader()):
         #喂入一个batch的测试数据
        test_cost= exe.run(program=test_program, feed=feeder.feed(data), fetch_list=[avg_cost])

    print('训练轮次:%5d  训练集损失值：%5.5f  测试集损失值:%5.5f' % (pass_id, train_cost[0][0], test_cost[0][0]))     #打印最后一个batch的损失值
    
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print ('save models to %s' % (model_save_dir))

title="training cost"
plt.title(title, fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("cost", fontsize=14)
plt.plot(iters, train_costs,color='red',label='training cost') 
plt.grid()
plt.show()

#保存训练参数到指定路径中，构建一个专门用预测的program, 保存推理model的路径
fluid.io.save_inference_model(model_save_dir, ['x'], [y_predict], exe)

infer_exe = fluid.Executor(place)    #创建推测用的executor
inference_scope = fluid.core.Scope() #Scope指定作用域

infer_results=[]
groud_truths=[]


with fluid.scope_guard(inference_scope):#修改全局/默认作用域（scope）, 运行时中的所有变量都将分配给新的scope。
    #从指定目录中加载 推理model(inference model)
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(model_save_dir, infer_exe)
    #获取预测数据
    infer_reader = paddle.batch(paddle.dataset.uci_housing.test(), batch_size=100)    #从测试数据中读取一个大小为100的batch数据

    #从test_reader中分割x
    test_data = next(infer_reader())
    test_x = np.array([data[0] for data in test_data]).astype("float32")
    test_y= np.array([data[1] for data in test_data]).astype("float32")
    results = infer_exe.run(inference_program,                              #预测模型
                            feed={feed_target_names[0]: np.array(test_x)},  #喂入要预测的x值
                            fetch_list=fetch_targets)                       #得到推测结果 
                            
    # print("infer results: (House Price)")
    for idx, val in enumerate(results[0]):
        # print("%d: %.2f" % (idx, val))
        infer_results.append(val)
    # print("ground truth:")
    for idx, val in enumerate(test_y):
        # print("%d: %.2f" % (idx, val))
        groud_truths.append(val)


    print(len(infer_results), len(groud_truths))

    plt.title('Boston', fontsize=24)
    x = range(0, 100)  # 设置x范围
    y = groud_truths  # 得到y的列表
    y1 = infer_results  # 得到y的列表
    plt.plot(x, y, color='g', label='groud_truths')
    plt.plot(x, y1, color='r', label='infer_results')

    plt.grid(True, linestyle='--', alpha=0.5)  # 显示函数网格
    plt.legend()  # 显示函数label需要这个函数
    plt.ylim(0, 38)  # 设置y轴刻度范围
    plt.yticks([0, 35], ['0', '35'])
    plt.ylabel('predict', y=1, rotation=0, fontsize=10)  # 若 y=0.2 表示label放在y轴的20%的位置处。
    plt.xlim(0, 20)  # 设置x轴刻度范围
    plt.xticks(x[::5])
    plt.xlabel('epoch', fontsize=10)
    plt.show()


    title='Boston'
    plt.title(title, fontsize=24)
    x = np.arange(1,32) 
    y = x
    plt.plot(x, y)
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('infer result', fontsize=14)
    plt.scatter(groud_truths, infer_results,color='green',label='training cost') 
    plt.grid()
    plt.show()