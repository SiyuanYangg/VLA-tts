import types

def new_specific(self):
    print("只对当前实例起作用")

class Base():
    def process(self):
        print("hhh")

obj = Base()
obj.process = types.MethodType(new_specific, obj)

obj.process()  # 只对 obj 起作用
other = Base()
other.process()  # 仍旧是原始逻辑
