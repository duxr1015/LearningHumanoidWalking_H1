import contextlib
import os
import numpy as np
import mujoco
from mujoco import viewer as mujoco_viewer

DEFAULT_SIZE = 500

class MujocoEnv():
    """
    Superclass for all MuJoCo environments.
    """

    def __init__(self,model, sim_dt, control_dt):
    
        self.model = self.model      #加载模型
        self.data = mujoco.MjData(self.model) #创建运行时数据对象
        self.viewer = None

        # set frame skip and sim dt
        self.frame_skip = (control_dt/sim_dt)
        self.model.opt.timestep = sim_dt

        self.init_qops = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

    # methods to override
    # -------------------

    

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass
        """
        raise NotImplementedError #需子类实现

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        self.viewer.cam.trackbodyid = 1  # ID从0开始，1代表跟踪第2个物体
        self.viewer.cam.distance = self.model.stat.extent * 1.5 # 设置相加与跟踪物体的距离，为模型特征长度的1.5倍，确保有适当的视野范围。
        self.viewer.cam.lookat[2] = 1.5
        self.viewer.cam.lookat[0] = 2.0  # 将焦点设置在(2.0,0,1.5),即机器人前方略高处
        self.viewer.cam.elevation = -20  # 相机俯仰角，-20为向下倾斜20度
        self.viewer.vopt.geomgroup[2] = 0 # 隐藏第3个几何组
        self.viewer._render_every_frame = True # 每帧都渲染

    def viewr_is_paused(self):
        return self.viewer._paused

    # -----------------------------
    # (some methods are taken directly from dm_control)

    @contextlib.contextmanager
    def disable(self,*flags):
        """Context manager for temporarily disabling MuJoCo flags.

        Args:
          *flags: Positional arguments specifying flags to disable. Can be either
            lowercase strings (e.g. 'gravity', 'contact') or `mjtDisableBit` enum
            values.

        Yields:
          None

        Raises:
          ValueError: If any item in `flags` is neither a valid name nor a value
            from `mujoco.mjtDisableBit`.
        """
        # 1. 保存原始的disableflags(位掩码)
        old_bitmask = self.model.opt.disableflags
        new_bitmask = old_bitmask # 基于原始值修改

        for flag in flags:
            if isinstance(flag, str):
                # 字符串转枚举 (如'actuation' -> mjDSBL_ACTUATION)
                try:
                    field_name = "mjDSBL_" + flag.upper()  # 拼接枚举名（MuJoCo 枚举命名规范）
                    flag = getattr(mujoco.mjtDisableBit, field_name)  # 修正枚举类型
                except AttributeError:
                    valid_names = [
                        field_name.split("_")[1].lower()   # 从"mjDSBL_ACTUATION"提取"actuation"
                        for field in mujoco.mjtDisableBit.__members__
                        if field.startswith("mjDSBL_")  # 过滤非禁用标志的枚举
                    ]
                    raise ValueError("'{}' is not a valid flag name. Valid names:{}"
                                     .format(flag,", ".join(valid_names))) from None
            elif isinstance(flag,int):
                # 整型转枚举(确保类型安全)
                flag = mujoco.mjtDisableBit(flag)
            else:
                 raise ValueError(f"Invalid flag type: {type(flag)}. Must be str or int.")
            new_bitmask |= flag.value # 按位或赋值操作，将对应位置为1(表示禁用该特性)
            
        self.model.opt.disableflags = new_bitmask
        try:
            yield # 执行 with 语句块内的代码
        finally:
            self.model.opt.disableflags = old_bitmask # 无论是否出错都恢复原始设置
        
    def reset(self):
        mujoco.mj_resetData(self.model,self.data) # 重置 MuJoCo 运行时数据
        ob = self.reset_model() # 调用子类实现的具体重置逻辑
        return ob
    
    def set_state(self,qpos,qvel):
        #1.校验输入形状(确保与模型自由度匹配)
        assert qpos.shape == (self.model.nq,), \
            f"qpos shape {qpos.shape} is expected to be {(self.model.nq,)}"
        
        assert qvel.shape == (self.model.nv,), \
            f"qvel shape {qvel.shape} is expected to be {(self.model.nv,)}"
        
        #2.设置位置和速度
        self.data.qpos[:] = qpos #赋值关节位置(nq 为位置自由度)
        self.data.qvel[:] = qvel #赋值关节速度(nv 为速度自由度)

        #3.清空驱动信号和插件状态(避免残留控制信号影响)
        self.data.act = [] #清空驱动信号(actuation)
        self.data.plugin_state = [] #清空插件状态(如自定义传感器数据)

        #4.临时禁用驱动，更新仿真状态
        with self.disable('actuation'): #禁用驱动功能(避免驱动信号干扰新状态)
            mujoco.mj_forward(self.model,self.data) # 基于新 qpos/qvel 计算其他状态（如加速度、约束力）

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip
    
    def render(self):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model,self.data) # 初始化可试器
            self.viewer_setup() # 配置相机参数
        self.viewer.render() # 渲染当前帧

    def uploadGPU(self, hfieldid=None, meshid=None, texid=None):
        # 上传高度场到GPU
        if hfieldid is not None:
            mujoco.mjr_uploadHField(self.model, self.viewer.ctx, hfieldid)
        # 上传网格到GPU
        if meshid is not None:
            mujoco.mjr_uploadMesh(self.model, self.viewer.ctx, meshid)
        # 上传纹理到GPU
        if texid is not None:
            mujoco.mjr_uploadTexture(self.model, self.viewer.ctx, texid)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()  # 关闭可视化窗口
            self.viewer = None  # 释放引用，便于垃圾回收




    
    
    