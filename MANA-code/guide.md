- 代码执行方法：

cd .../MANA-code
python 【python程序的相对路径】 【修改参数】

- 具体文件夹的说明：

GY-p2/
    data/
        存放了在GY-p2数据的day24上跑所需要的所有数据。GY的所有程序都只能跑--variable_num 24（已经设置为默认参数）。

    daily-decoding/     （之前已经验证过了）
        单天解码
        --mix_length 28
    long-term-decoding/ （之前已经验证过了）
        跨天解码
        --mix_length 28
    CL-MA-AD/           （之前已经验证过了）
        不同部分的消融实验
    training-weeks/
        训练使用数据的周数（Fig2b2）
        --train_weeks 【1,2,3,5】 （4就是long-term-decoding）
    AD-1-model/
        只训练一个解码模型的对比实验（FigEx3）

AI-tasks/
    data/
        存放了在AI-tasks上跑所需要的所有数据。参数可以设置--variable 【5~12】（默认为5）。

    dvsgesture/         （之前已经验证过了）
    drift/              （之前已经验证过了）
    rotate/             （之前已经验证过了）

Jango/
    data/
        存放了在Jango上跑所需要的所有数据。参数可以设置--variable_num 【6~20】（默认为6）。

    long-term-decoding/
        跨天解码