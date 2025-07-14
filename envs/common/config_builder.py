import os
import yaml
from typing import Any, Dict, Optional, Union, List

class Configuration:
    """A class to handle configuration data with attribute-style access."""
    #构造方法支持嵌套字典的递归转换
    def __init__(self,**kwargs: Any) ->None:
        #智能类型处理：
        for key, value in kwargs.items():
            if isinstance(value,dict):
                setattr(self,key,Configuration(**value))  # 递归转换嵌套字典
            elif isinstance(value, list) and all(isinstance(item,dict) for item in value):
                setattr(self,key,[Configuration(**item) for item in value])  # 处理列表中的字典
            else:
                setattr(self,key,value)  # 原始值直接赋值

    def __repr__(self) -> str:
        """Return a string representation of the configuration."""
        return str(self.__dict__)
    
    def __getattr__(self,name:str) -> None:
        """Return None for non-existent attributes instead of raising an error."""
        return None
    
    def to_dict(self) -> Dict[str,Any]:
        """Convert the Configuration object back to a dictionary."""
        result={}
        for key,value in self.__dict__.items():
            if isinstance(value,Configuration):
                result[key] = value.to_dict()   # 递归转换子对象
            elif isinstance(value,list) and value and isinstance(value[0],Configuration):
                result[key] = [item.to_dict() if isinstance(item,Configuration) else item for item in value] # 处理列表中的子对象
            else:
                result[key] = value  # 原始值直接保存
        return result
    
def load_yaml(file_path:str) -> Configuration:
    """
    Load configuration from a YAML file.
    Args:
        file_path: Path to the YAML file.
    Returns:
        Configuration object with data from the YAML file.
    Raises:
        FileNotFoundError: If the file doesn't exist.
        yaml.YAMLError: If there's an error parsing the YAML.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found:{file_path}")
    
    with open(file_path,'r') as file:
        try:
            config_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file{file_path}:{e}")
    return Configuration(**config_data)    
            


############### test ################
if __name__ == "__main__":
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接配置文件的完整路径，确保路径正确
    yaml_file_path = os.path.join(current_dir, "test.yaml")  
    
    try:
        config = load_yaml(yaml_file_path)
        # 属性式访问
        print(config.training.epochs)  
        print(config.optimizer.lr)     

        # 安全访问不存在的属性
        print(config.non_existent_key)  

        # 修改配置
        config.training.epochs = 200

        # 转回字典
        config_dict = config.to_dict()
        print(config_dict["model"]["layers"])  
    except FileNotFoundError as e:
        print(f"加载配置文件失败: {e}")
    except yaml.YAMLError as e:
        print(f"解析YAML文件失败: {e}")
    except AttributeError as e:
        print(f"访问配置属性失败: {e}")
