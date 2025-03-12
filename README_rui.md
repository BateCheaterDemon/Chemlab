# 安装

大概的思路

采集数据需要安装 isaac sim 4.1 版本，训练需要再安装 diffusion policy 的环境

记得conda新建一个环境

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 # TODO: change this to your cuda version
pip install isaacsim==4.1.0 isaacsim-extscache-physics==4.1.0 isaacsim-extscache-kit==4.1.0 isaacsim-extscache-kit-sdk==4.1.0 --extra-index-url https://pypi.nvidia.com
```

# 数据采集

修改 `config` 下配置文件的参数

```shell
python table.py
```

# 训练

修改 `diffusion_policy/config` 下的配置文件

```shell
python train.py --config-name=aaa
```

# 推理测试

修改代码中的文件路径

```shell
python inference.py
```