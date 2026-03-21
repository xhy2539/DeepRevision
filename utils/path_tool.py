#为整个工程提供统一绝对路径
import os.path


def get_project_root()->str:
    #当前文件绝对路径
    current_file=os.path.abspath(__file__)
    #获取工程根目录
    current_dir=os.path.dirname(current_file)

    project_root=os.path.dirname(current_dir)

    return project_root

def get_abs_path(relative_path:str)->str:
    #输入相对路径返回绝对路径
    project_root=get_project_root()
    return os.path.join(project_root,relative_path)