# Neural-network-based-phase-recovery-of-highly-random-laser-arrays
Neural network-based phase recovery of highly random laser arrays
权重文件在huggingface上：Lukaipeng/Neural_network_based_phase_recovery
环境安装，如果您有自己的pytorch环境，直接使用即可，可能会需要下载一下scipy，opencv，PIL（如果您的环境中没有的话）

使用方式：
训练阶段：在train.py中设置好预训练模型，如果不想加载预训练模型，则将加载模型的代码注释掉即可，设置好环境和路径后即可点击运行，若要从0开始训练，大概一个星期能收敛（代码中判别器的损失函数设计来源pix2pix）

验证训练情况：自行设计，如果只是单纯看效果，那neu_out提供了从目标图（需要自己制作，也可以使用data_generate输出）恢复到相位图的方法

此训练方案，实现了5*5，每个点有9个自由度的相位恢复，相比同类型恢复相位，其恢复复杂程度达到SOTA，是其他同类模型不可比拟的。但由于本人知识和计算资源受限，因此希望有人能在此基础上做出自由度更，分辨率更高的基于神经网络恢复激光阵列点相位。
