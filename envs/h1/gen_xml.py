import os
import numpy as np
import models
from dm_control import mjcf
import transforms3d as tf3

#原始模型路径：Unitree H1 机器人的MuJoCo XML模型
H1_DESCRIPTION_PATH = os.path.join(os.path.dirname(models.__file__),"unitree_h1/scene.xml")

#关节分类(便于后续筛选或修改)
LEG_JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
              "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"] # 腿部关节
WAIST_JOINTS = ["torso"]  # 腰部关节
ARM_JOINTS = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
              "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"] # 手臂关节


def create_rangefinder_array(mjcf_model, num_rows=4,num_cols=4,spacing=0.4):
    for i in range(num_rows*num_cols):
        name = 'rf' + repr(i) #传感器名称
        # 计算传感器在骨盆坐标系中的位置(形成4×4网格)
        u = (i % num_cols)
        v = (i // num_cols)
        x = (v - (num_cols - 1) / 2) * spacing # x坐标 左右方向
        y = ((num_rows - 1) / 2 - u) * (-spacing) #y坐标 前后方向

        #1. 添加标记点(site)：传感器的物理位置
        mjcf_model.find('body','pelvis').add('site',
                                             name = name,
                                             pos=[x,y,0], # 高度为0（骨盆平面）
                                             quat = '0 1 0 0') # 朝向（沿y轴向前）
        #2. 添加测距传感器：关联到上述site
        mjcf_model.sensor.add('rangefinder',
                              name = name,
                              site = name) # 传感器绑定到site位置
    
    return mjcf_model

def remove_joints_and_actuators(mjcf_model, config):
    # 1. 移除配置中指定的未使用关节
    for limb in config['unused_joints']:
        for joint in limb:
            mjcf_model.find('joint', joint).remove()  # 从mjcf模型中删除关节
    
    # 2. 移除没有对应关节的执行器（电机）
    for mot in mjcf_model.actuator.motor:
        mot.user = None  # 清除用户数据
        if mot.joint is None:  # 若电机未绑定关节（关节已被移除）
            mot.remove()  # 删除电机
    return mjcf_model


def builder(export_path,config):
    print("Modifying XML model...")
    print(f"导出路径: {export_path}")
    print(f"H1_DESCRIPTION_PATH: {H1_DESCRIPTION_PATH}")  # 添加这行
    #1. 从原始路径加载H1机器人的mjcf模型
    # 确保导出目录存在
    os.makedirs(export_path, exist_ok=True)

    # 加载原始XML
    mjcf_model = mjcf.from_path(H1_DESCRIPTION_PATH)
    mjcf_model.model = 'h1' #设置模型名称

    #2. 移除未使用的关节和执行器(简化模型)
    mjcf_model = remove_joints_and_actuators(mjcf_model,config)

    #3. 配置视觉和碰撞几何的分组(用于渲染控制)
    mjcf_model.find('default','visual').geom.group = 1  # 视觉几何→组1
    mjcf_model.find('default','collision').geom.group = 2  # 碰撞几何→组2
    # （注：MuJoCo中，geom.group用于控制渲染时的显示/隐藏，如隐藏组2的碰撞体）

    #4. 配置关节和电机的限制属性(从config读取)
    if 'ctrllimited' in config:
        # 设置电机是否启用控制范围限制
        mjcf_model.find('default','h1').motor.ctrllimited = config['ctrllimited']
    if 'jointlimited' in config:
        # 设置关节是否启用运动范围限制
        mjcf_model.find('default', 'h1').joint.limited = config['jointlimited']

    # 5. 重命名所有电机，符合命名约定（关节名+"_motor"）
    for mot in mjcf_model.actuator.motor:
        mot.name = mot.name + "_motor"  # 如"left_hip_yaw"→"left_hip_yaw_motor"

    # 6. 移除关键帧（初始姿态定义），避免干扰自定义初始化
    mjcf_model.keyframe.remove()

    # 7. 若启用"minimal"模式，进一步简化模型（仅保留碰撞体）
    if 'minimal' in config and config['minimal']:
        mjcf_model.find('default', 'collision').geom.group = 1  # 碰撞体→组1（便于显示）
        # 移除视觉网格和几何（只保留碰撞体，加速仿真）
        meshes = mjcf_model.asset.mesh
        for mesh in meshes:
            mesh.remove()  # 删除网格资产
        for geom in mjcf_model.find_all('geom'):
            if geom.dclass and geom.dclass.dclass == "visual":
                geom.remove()  # 删除视觉几何

    # 8. 设置骨盆的自由关节名称为"root"（便于后续引用根节点）
    mjcf_model.find('body', 'pelvis').freejoint.name = 'root'

    # 9. 若配置启用，添加测距传感器阵列
    if 'rangefinder' in config and config['rangefinder']:
        mjcf_model = create_rangefinder_array(mjcf_model)

    # 10. 若配置启用，添加抬高的平台（用于测试机器人爬坡/跨越能力）
    if 'raisedplatform' in config and config['raisedplatform']:
        block_pos = [2.5, 0, 0]  # 平台位置（世界坐标系）
        block_size = [3, 3, 1]   # 平台尺寸（x,y,z）
        name = 'raised-platform'
        # 添加平台体到世界
        mjcf_model.worldbody.add('body', name=name, pos=block_pos)
        # 为平台添加碰撞几何
        mjcf_model.find('body', name).add('geom', name=name, group='3',
                                          condim='3', friction='.8 .1 .1',  # 摩擦系数
                                          size=block_size, type='box', material='')

    # 11. 配置模型大小参数（-1表示使用默认最大值）
    mjcf_model.size.njmax = "-1"  # 最大关节数
    mjcf_model.size.nconmax = "-1"  # 最大碰撞数
    mjcf_model.size.nuser_actuator = "-1"  # 最大用户定义执行器数

    # 12. 导出模型（包含资产文件）到指定路径
    mjcf.export_with_assets(mjcf_model, out_dir=export_path, precision=5)
    path_to_xml = os.path.join(export_path, mjcf_model.model + '.xml')
    print("Exporting XML model to ", path_to_xml)
    return


