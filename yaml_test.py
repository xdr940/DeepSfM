import yaml


class YamlHandler:
    def __init__(self, file):
        def dataset():
            self.a = 'bbb'
            self.c='ccc'
        def model():
            self.e='eee'
        dataset()
        model()
        self.file = file

    def read_yaml(self, encoding='utf-8'):
        """读取yaml数据"""
        with open(self.file, encoding=encoding) as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    def write_yaml(self, data, encoding='utf-8'):
        """向yaml文件写入数据"""
        with open(self.file, encoding=encoding, mode='w') as f:
            return yaml.dump(data, stream=f, allow_unicode=True)


if __name__ == '__main__':

    # 读取config.yaml配置文件数据
    args = YamlHandler('./opts/train_opts.yaml').read_yaml()
    print(args)


    # 将data数据写入config1.yaml配置文件
    write_data = YamlHandler('./train_mc2.yaml').write_yaml(args)
    print(args)