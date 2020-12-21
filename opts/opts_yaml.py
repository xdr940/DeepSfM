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