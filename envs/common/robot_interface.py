import os
import numpy as np
import transforms3d as tf3
import mujoco
import torch
import collections

class RobotInterface(object):
    def __init__(self,model,data,rfoot_body_name=None,lfoot_body_name=None,
                 path_to_nets=None):
        self.model = model # MuJoCo 模型定义(静态结构)
        self.data = data   # MuJoCo 运行时数据(动态状态)

        # 关键部件名称(用于后续状态获取和控制)
        self.rfoot_body_name = rfoot_body_name #右脚部件名称
        self.lfoot_body_name = lfoot_body_name #左脚部件名称
        self.floor_body_name = model.body(0).name # 地面名称（通常是第一个 body）
        self.robot_root_name = model.body(1).name # 机器人根节点名称（通常是第二个 body）

        self.stepCounter = 0   # 仿真步数计数器(用于控制频率或记录时间)

    def load_motor_nets(self, path_to_nets):
        self.motor_dyn_nets = {}  # 存储各关节的电机动态模型

        # 遍历指定路径下的所有文件夹(每个文件夹对应一个关节)
        for jnt in os.listdir(path_to_nets):
            if not os.path.isdir(os.path.join(path_to_nets,jnt)):
                continue

            # 加载预训练的 TorchScript 模型
            net_path = os.path.join(path_to_nets,jnt,"trained_jit.pth")
            net = torch.jit.load(net_path)
            net.eval() # 设置为评估模式(关闭训练相关操作，如dropout)
            self.motor_dyn_nets[jnt] = net # 按关节名称存储模型

        # 创建滑动窗口缓冲区(用于存储历史数据，支持时序预测)
        self.ctau_buffer = collections.deque(maxlen=25)  # 控制力矩历史
        self.qdot_buffer = collections.deque(maxlen=25)  # 关节速度历史

    def motor_nets_forward(self,cmdTau):
        #1.初始化阶段(缓冲区未满时)
        if len(self.ctau_buffer) < self.ctau_buffer.maxlen:
            w = self.get_act_joint_velocities()
            self.qdot_buffer.append(w)
            self.ctau_buffer.append(cmdTau)
            return cmdTau
        #2.周期性数据更新
        if (self.stepCounter % 2) == 0:
            w = self.get_act_joint_velocities()
            self.qdot_buffer.append(w)
            self.ctau_buffer.append(cmdTau)

        #3.神经网络推理
        actTau = np.copy(cmdTau)
        for jnt in self.motor_dyn_nets.keys():
            # 获取关节对应的执行器ID
            jnt_id = mujoco.mj_name2id(self.model,mujoco.mjObj.mjOBJ_ACTUATOR,jnt + '_motor')
            nn = self.motor_dyn_nets[jnt].double()  # 获取对应关节的神经网络模型

            # 准备输入：拼接历史关节速度和控制信号
            qdot = torch.tensor(np.array(self.qdot_buffer),dtype=torch.double)
            ctau = torch.tensor(np.array(self.ctau_buffer), dtype=torch.double)
            inp = torch.cat([qdot[:, jnt_id], ctau[:, jnt_id]])

            # 模型推理：预测实际关节力矩
            actTau[jnt_id] = nn(inp)
        return actTau
    
    ##########模型静态属性##########

    # 位置自由度数量
    def nq(self):
        return self.model.nq
    
    # 控制自由度数量
    def nu(self):
        return self.model.nu

    # 速度自由度数量
    def nv(self):
        return self.model.nv
    
    # 仿真时间步长
    def sim_dt(self):
        return self.model.opt.timestep

    # 机器人总质量
    def get_robot_mass(self):
        return mujoco.mj_getTotalmass(self.model)
    

    ###############数据动态属性##########

    # 关节位置
    def get_qpos(self):
        return self.data.qpos.copy()

    # 关节速度
    def get_qvel(self):
        return self.data.qvel.copy()

    # 关节加速度
    def get_qacc(self):
        return self.data.qacc.copy()

    # 接触点速度
    def get_cvel(self):
        return self.data.cvel.copy()


    ###############关节与力信息#############

    # 关节 ID 查询
    def get_jnt_id_by_name(self, name):
        return self.model.joint(name)

    # 关节位置索引
    def get_jnt_qposadr_by_name(self, name):
        return self.model.joint(name).qposadr

    # 关节速度索引
    def get_jnt_qveladr_by_name(self, name):
        return self.model.joint(name).dofadr
    # 外部接触力
    def get_body_ext_force(self):
        return self.data.cfrc_ext.copy()
    
    
    # 获取电机速度限制(rad/s)
    def get_motor_speed_limits(self):
        rpm_limits = self.model.actuator_user[:,0] #从模型中获取RPM限制
        return ((rpm_limits)*(2*np.pi/60)) #转换为 rad/s
    
    # 获取关节速度限制(rad/s)
    def get_act_joint_speed_limits(self):
        gear_rations = self.model.actuator_gear[:,0] #获取传动比
        mot_lims = self.get_motor_speed_limits() #获取电机速度限制
        return [float(i/j) for i,j in zip(mot_lims,gear_rations)] #关节速度=电机速度/传动比
    
    # 获取传动比
    def get_gear_rations(self):
        return self.model.actuator_gear[:,0]
    
    # 获取电机名称
    def get_motor_names(self):
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) 
                          for i in range(self.model.nu)]
        return actuator_names

    # 获取驱动关节的索引
    def get_actuated_joint_inds(self):
        """
        Returns list of joint indices to which actuators are attached.
        """
        # 获取所有的关节名称(按ID排序)
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        # 获取所有执行器的名称(按ID排序)
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        # 筛选：关节名称 + "_motor" 存在于执行器名称中-> 该关节被驱动
        return [idx for idx, jnt in enumerate(joint_names) if jnt+'_motor' in actuator_names]

    # 获取驱动关节的名称
    def get_actuated_joint_names(self):
        """
        Returns list of joint names to which actuators are attached.
        """
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        return [jnt for idx, jnt in enumerate(joint_names) if jnt+'_motor' in actuator_names]

    # 获取驱动关节在qpos中的索引
    def get_motor_qposadr(self):
        """
        Returns the list of qpos indices of all actuated joints.
        """
        indices = self.get_actuated_joint_inds() # 驱动关节的索引
        return [self.model.jnt_qposadr[i] for i in indices] # 对应qpos中的位置

    ##############电机层面状态查询###############
    # 电机位置
    def get_motor_positions(self):
        """
        Returns position of actuators.
        """
        return self.data.actuator_length   # 电机输出轴的位移/转角（单位：米或弧度）

    # 电机速度
    def get_motor_velocities(self):
        """
        Returns velocities of actuators.
        """
        return self.data.actuator_velocity  # 电机输出轴的速度（单位：弧度/秒或米/秒）

    ######## 电机-关节状态转换(核心功能)##########
    # 关节层面的力矩
    def get_act_joint_torques(self):
        """
        Returns actuator force in joint space.
        """
        gear_ratios = self.model.actuator_gear[:,0] # 齿轮比（>1表示减速，力矩放大）
        motor_torques = self.data.actuator_force # 电机输出力矩
        return (motor_torques*gear_ratios)  # 关节力矩 = 电机力矩 × 齿轮比

    # 关节层面的位置
    def get_act_joint_positions(self):
        """
        Returns position of actuators at joint level.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        motor_positions = self.get_motor_positions() # 电机位置
        return (motor_positions/gear_ratios) # 关节位置 = 电机位置 / 齿轮比
    
    # 关节层面的速度
    def get_act_joint_velocities(self):
        """
        Returns velocities of actuators at joint level.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        motor_velocities = self.get_motor_velocities()
        return (motor_velocities/gear_ratios)

    #############单个关节的状态查询###############
    # 单个关节的位置
    def get_act_joint_position(self, act_name):
        """
        Returns position of actuator at joint level.
        """
        assert len(self.data.actuator(act_name).length)==1
        # 关节位置 = 电机位置 / 该执行器的齿轮比
        return self.data.actuator(act_name).length[0]/self.model.actuator(act_name).gear[0]

    # 单个关节的速度
    def get_act_joint_velocity(self, act_name):
        """
        Returns velocity of actuator at joint level.
        """
        assert len(self.data.actuator(act_name).velocity)==1
        return self.data.actuator(act_name).velocity[0]/self.model.actuator(act_name).gear[0]

    #####################关节运动范围查询#####################
    # 所有驱动关节的限位
    def get_act_joint_ranges(self):
        """
        Returns the lower and upper limits of all actuated joints.
        """
        indices = self.get_actuated_joint_inds()  # 驱动关节索引
        low, high = self.model.jnt_range[indices, :].T # 提取限位
        return low, high  # 下限数组和上限数组

    # 单个关节的限位
    def get_act_joint_range(self, act_name):
        """
        Returns the lower and upper limits of given joint.
        """
        low, high = self.model.joint(act_name).range  # 直接获取关节的range属性
        return low, high
    

    #################控制与传感器数据获取###########
    # 获取执行器控制范围
    def get_actuator_ctrl_range(self):
        """
        Returns the acutator ctrlrange defined in model xml.
        """
        low, high = self.model.actuator_ctrlrange.copy().T
        return low, high

    # 获取执行器自定义数据
    def get_actuator_user_data(self):
        """
        Returns the user data (if any) attached to each actuator.
        """
        return self.model.actuator_user.copy()

    def get_root_body_pos(self):
        return self.data.xpos[1].copy()

    def get_root_body_vel(self):
        qveladr = self.get_jnt_qveladr_by_name("root")
        return self.data.qvel[qveladr:qveladr+6].copy()

    # 获取传感器数据
    def get_sensordata(self, sensor_name):
        sensor = self.model.sensor(sensor_name)
        sensor_adr = sensor.adr[0]
        data_dim = sensor.dim[0]
        return self.data.sensordata[sensor_adr:sensor_adr+data_dim]

    def get_rfoot_body_pos(self):
        if isinstance(self.rfoot_body_name, list):
            return [self.data.body(i).xpos.copy() for i in self.rfoot_body_name]
        return self.data.body(self.rfoot_body_name).xpos.copy()

    def get_lfoot_body_pos(self):
        if isinstance(self.lfoot_body_name, list):
            return [self.data.body(i).xpos.copy() for i in self.lfoot_body_name]
        return self.data.body(self.lfoot_body_name).xpos.copy()

    ############# 接触检测与地面反作用力计算（核心功能）
    # 检测特定部件与地面的接触
    def get_body_floor_contacts(self, body_name):
        """
        Returns list of 'body' and floor contacts.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]  # 获取所有接触
        body_contacts = []

        body_names = [body_name] if isinstance(body_name, str) else body_name
        body_ids = [self.model.body(bn).id for bn in body_names]
        for i,c in enumerate(contacts):
            # 判断接触的两个几何体是否分别属于地面和目标部件
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            geom1_is_floor = (self.model.body(geom1_body.rootid).name!=self.robot_root_name)
            geom2_is_body = (self.model.geom_bodyid[c.geom2] in body_ids)
            if (geom1_is_floor and geom2_is_body):
                body_contacts.append((i,c))
        return body_contacts

    def get_rfoot_floor_contacts(self):
        """
        Returns list of right foot and floor contacts.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        rcontacts = []

        rfeet = [self.rfoot_body_name] if isinstance(self.rfoot_body_name, str) else self.rfoot_body_name
        rfeet_ids = [self.model.body(bn).id for bn in rfeet]
        for i,c in enumerate(contacts):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            geom1_is_floor = (self.model.body(geom1_body.rootid).name!=self.robot_root_name)
            geom2_is_rfoot = (self.model.geom_bodyid[c.geom2] in rfeet_ids)
            if (geom1_is_floor and geom2_is_rfoot):
                rcontacts.append((i,c))
        return rcontacts

    def get_lfoot_floor_contacts(self):
        """
        Returns list of left foot and floor contacts.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        lcontacts = []

        lfeet = [self.lfoot_body_name] if isinstance(self.lfoot_body_name, str) else self.lfoot_body_name
        lfeet_ids = [self.model.body(bn).id for bn in lfeet]
        for i,c in enumerate(contacts):
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            geom1_is_floor = (self.model.body(geom1_body.rootid).name!=self.robot_root_name)
            geom2_is_lfoot = (self.model.geom_bodyid[c.geom2] in lfeet_ids)
            if (geom1_is_floor and geom2_is_lfoot):
                lcontacts.append((i,c))
        return lcontacts
    
    #计算足部地面反作用力（GRF）
    def get_rfoot_grf(self):
        """
        Returns total Ground Reaction Force on right foot.
        """
        right_contacts = self.get_rfoot_floor_contacts()
        rfoot_grf = 0
        for i, con in right_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            rfoot_grf += np.linalg.norm(c_array)
        return rfoot_grf

    def get_lfoot_grf(self):
        """
        Returns total Ground Reaction Force on left foot.
        """
        left_contacts = self.get_lfoot_floor_contacts()
        lfoot_grf = 0
        for i, con in left_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            lfoot_grf += (np.linalg.norm(c_array))
        return lfoot_grf
    

    # 计算作用于指定部件的总接触力
    def get_body_contact_force(self, body):
        """
        Returns total contact force acting on a body (or list of bodies).
        """
        if isinstance(body, str):
            body = [body] # 支持单个部件名或列表
        frc = 0 # 总接触力初始化
        for i, con in enumerate(self.data.contact):
            # 获取接触点的6维力向量（3力+3力矩）
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            # 确定接触双方的部件
            b1 = self.model.body(self.model.geom(con.geom1).bodyid)
            b2 = self.model.body(self.model.geom(con.geom2).bodyid)
            # 若接触涉及目标部件，累加接触力的模长
            if b1.name in body or b2.name in body:
                frc += np.linalg.norm(c_array) # 力向量的模长（总力大小）
        return frc

    #计算两个部件间的交互力
    def get_interaction_force(self, body1, body2):
        """
        Returns contact force beween a body1 and body2.
        """
        frc = 0
        for i, con in enumerate(self.data.contact):
            # 获取接触力向量
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            # 确定接触双方的部件
            b1 = self.model.body(self.model.geom(con.geom1).bodyid)
            b2 = self.model.body(self.model.geom(con.geom2).bodyid)
            # 若接触双方为目标部件，累加接触力模长
            if (b1.name==body1 and b2.name==body2) or (b1.name==body2 and b2.name==body1):
                frc += np.linalg.norm(c_array)
        return frc

    ###############部件速度查询##############
    # 获取指定部件的速度
    def get_body_vel(self, body_name, frame=0):
        """
        Returns translational and rotational velocity of a body in body-centered frame, world/local orientation.
        """
        body_vel = np.zeros(6) # 存储6维速度（3角速度+3线速度）
        # 获取部件ID
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        # 调用MuJoCo内置函数计算速度
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                 body_id, body_vel, frame)
        return [body_vel[3:6], body_vel[0:3]] # [角速度, 线速度]

    #获取右脚速度
    def get_rfoot_body_vel(self, frame=0):
        """
        Returns translational and rotational velocity of right foot.
        """
        if isinstance(self.rfoot_body_name, list):
            return [self.get_body_vel(i, frame=frame) for i in self.rfoot_body_name]
        return self.get_body_vel(self.rfoot_body_name, frame=frame)
    #获取左脚速度
    def get_lfoot_body_vel(self, frame=0):
        """
        Returns translational and rotational velocity of left foot.
        """
        if isinstance(self.lfoot_body_name, list):
            return [self.get_body_vel(i, frame=frame) for i in self.lfoot_body_name]
        return self.get_body_vel(self.lfoot_body_name, frame=frame)
    
    # 根据物体名称和类型，获取物体在世界坐标系中的位置 xpos
    def get_object_xpos_by_name(self, object_name, object_type):
        if object_type=="OBJ_BODY":
            return self.data.body(object_name).xpos # 从"体"中直接获取位置（MuJoCo的body数据结构包含xpos字段）
        elif object_type=="OBJ_GEOM":
            return self.data.geom(object_name).xpos # 从"几何"中直接获取位置（MuJoCo的geom数据结构包含xpos字段）
        elif object_type=="OBJ_SITE":
            return self.data.site(object_name).xpos # 从"标记点"中获取位置（site数据结构包含xpos字段）
        else:
            raise Exception("object type should either be OBJ_BODY/OBJ_GEOM/OBJ_SITE.") # 类型不合法时抛出异常，明确允许的类型
    # 获取物体在世界坐标系中的旋转四元数 xquat
    def get_object_xquat_by_name(self, object_name, object_type):
        if object_type=="OBJ_BODY":
            return self.data.body(object_name).xquat # 体（body）直接存储四元数xquat，可直接返回
        if object_type=="OBJ_GEOM":
            # 几何（geom）存储的是旋转矩阵xmat，需转换为四元数
            xmat = self.data.geom(object_name).xmat  # 获取3x3旋转矩阵
            return tf3.quaternions.mat2quat(xmat)    # 调用transforms3d将矩阵转四元数
        if object_type=="OBJ_SITE":
            # 标记点（site）同样存储旋转矩阵xmat，转换为四元数
            xmat = self.data.site(object_name).xmat
            return tf3.quaternions.mat2quat(xmat)
        else:
            raise Exception("object type should be OBJ_BODY/OBJ_GEOM/OBJ_SITE.")

    # 生成物体的仿射变换矩阵（包含平移和旋转），用于坐标变换
    def get_object_affine_by_name(self, object_name, object_type):
        """Helper to create transformation matrix from position and quaternion."""
        """
        生成物体的仿射变换矩阵（包含平移和旋转），用于坐标变换。

        参数：
            object_name (str)：物体名称。
            object_type (str)：物体类型（同前）。

        返回：
            np.ndarray：4x4仿射矩阵，形式为：
                [
                    [R00, R01, R02, T0],
                    [R10, R11, R12, T1],
                    [R20, R21, R22, T2],
                    [0,   0,   0,   1]
                ]
            其中R是旋转矩阵，T是平移向量（世界坐标系下的位置）。
        """
        # 获取物体位置（平移分量）
        pos = self.get_object_xpos_by_name(object_name, object_type)
        # 获取物体四元数（旋转分量）
        quat = self.get_object_xquat_by_name(object_name, object_type)
        # 组合平移、旋转和缩放（缩放设为全1，即无缩放）生成仿射矩阵
        return tf3.affines.compose(pos, tf3.quaternions.quat2mat(quat), np.ones(3))

    def get_robot_com(self):
        """
        Returns the center of mass of subtree originating at root body
        i.e. the CoM of the entire robot body in world coordinates.
        """
        """
        获取机器人整体（以根体为起点的子树）的质心（CoM）在世界坐标系中的位置。

        返回：
            np.ndarray：长度为3的数组，代表质心坐标（x, y, z）。

        异常：
            若模型中未定义"subtreecom"传感器，则抛出异常。
        """
        # 获取所有传感器名称（从MuJoCo模型中遍历传感器ID转换为名称）
        sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)]
        # 检查是否存在"subtreecom"传感器（MuJoCo中用于测量子树质心
        if 'subtreecom' not in sensor_names:
            raise Exception("subtree_com sensor not attached.")
        # 返回根体子树的质心（索引1通常对应根体，0可能对应世界坐标系）
        return self.data.subtree_com[1].copy()

    def get_robot_linmom(self):
        """
        Returns linear momentum of robot in world coordinates.
        """
        """
        获取机器人整体在世界坐标系中的线性动量。

        返回：
            np.ndarray：长度为3的数组，代表线性动量（px, py, pz），单位为kg·m/s。

        异常：
            若模型中未定义"subtreelinvel"传感器，则抛出异常。
        """
        # 获取所有传感器名称
        sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)]
        # 检查是否存在"subtreelinvel"传感器（测量子树线速度）
        if 'subtreelinvel' not in sensor_names:
            raise Exception("subtree_linvel sensor not attached.")
        # 获取根体子树的线速度（质心处的线速度）
        linvel = self.data.subtree_linvel[1].copy()
        # 获取机器人总质量（需通过其他方法实现，如遍历所有体的质量求和）
        total_mass = self.get_robot_mass()
        # 线性动量 = 总质量 × 质心线速度
        return linvel*total_mass

    def get_robot_angmom(self):
        """
        Return angular momentum of robot's CoM about the world origin.
        """
        """
        获取机器人质心相对于世界坐标系原点的角动量。

        返回：
            np.ndarray：长度为3的数组，代表角动量（Lx, Ly, Lz），单位为kg·m²/s。

        异常：
            若模型中未定义"subtreeangmom"传感器，则抛出异常。
        """
        # 获取所有传感器名称
        sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)]
        # 检查是否存在"subtreeangmom"传感器（测量子树角动量)
        if 'subtreeangmom' not in sensor_names:
            raise Exception("subtree_angmom sensor not attached.")
        # 获取质心位置和线性动量
        com_pos = self.get_robot_com()
        lin_mom = self.get_robot_linmom()
        # 总角动量 = 子树绕质心的角动量 + 质心平移产生的角动量（科里奥利项
        return self.data.subtree_angmom[1] + np.cross(com_pos, lin_mom)
    

    # 足部碰撞检测
    def check_rfoot_floor_collision(self):
        """
        Returns True if there is a collision between right foot and floor.
        """
        return (len(self.get_rfoot_floor_contacts())>0)

    def check_lfoot_floor_collision(self):
        """
        Returns True if there is a collision between left foot and floor.
        """
        return (len(self.get_lfoot_floor_contacts())>0)

    # 异常碰撞检测 
    def check_bad_collisions(self, body_names=[]):
        """
        Returns True if there are collisions other than specifiedbody-floor,
        or feet-floor if body_names is not provided.
        """
        num_cons = 0
        if not isinstance(body_names, list):
            raise TypeError(f"expected list of body names, got '{type(body_names).__name__}'")
        # 若未指定body_names，默认检查双足与地面的碰撞
        if not len(body_names):
            num_rcons = len(self.get_rfoot_floor_contacts())
            num_lcons = len(self.get_lfoot_floor_contacts())
            num_cons = num_rcons + num_lcons
        # 统计指定部件与地面的碰撞数
        for bn in body_names:
            num_cons += len(self.get_body_floor_contacts(bn))
        # 若总碰撞数不等于指定碰撞数，说明存在其他异常碰撞
        return num_cons != self.data.ncon

    # 自碰撞检测
    def check_self_collisions(self):
        """
        Returns True if there are collisions other than any-geom-floor.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        for i,c in enumerate(contacts):
            # 获取碰撞双方的部件
            geom1_body = self.model.body(self.model.geom_bodyid[c.geom1])
            geom2_body = self.model.body(self.model.geom_bodyid[c.geom2])
            # 判断是否均为机器人部件（非地面）
            geom1_is_robot = self.model.body(geom1_body.rootid).name==self.robot_root_name
            geom2_is_robot = self.model.body(geom2_body.rootid).name==self.robot_root_name
            # 若两个几何体均属于机器人，则判定为自碰撞
            if geom1_is_robot and geom2_is_robot:
                return True
        return False
    

    def set_pd_gains(self, kp, kv):
        """
        设置PD控制器的比例增益（kp）和微分增益（kv）。

        参数：
            kp (np.ndarray)：比例增益数组，长度必须等于控制自由度（self.model.nu）。
            kv (np.ndarray)：微分增益数组，长度必须等于控制自由度（self.model.nu）。

        功能：
            - 验证增益数组的维度是否与执行器数量匹配（通过assert确保）。
            - 保存增益的副本（避免外部修改影响内部状态）。
        """
        # 断言：确保kp和kv的长度等于执行器数量（控制自由度）
        assert kp.size==self.model.nu
        assert kv.size==self.model.nu
        # 保存增益的副本（防止外部数组修改导致内部增益变化）
        self.kp = kp.copy()
        self.kv = kv.copy()
        return

    def step_pd(self, p, v):
        """
        执行PD控制计算，根据目标位置和速度与当前状态的误差，输出控制力矩。

        参数：
            p (np.ndarray)：目标位置数组（长度等于控制自由度）。
            v (np.ndarray)：目标速度数组（长度等于控制自由度）。

        返回：
            np.ndarray：计算得到的控制力矩数组（长度等于控制自由度）。

        功能：
            - 计算位置误差（当前位置与目标位置的差）和速度误差（当前速度与目标速度的差）。
            - 基于PD增益和误差，计算控制力矩（力矩 = kp×位置误差 + kv×速度误差）。
        """
        target_angles = p  # 目标位置（关节角度）
        target_speeds = v  # 目标速度（关节角速度）

        # 断言：确保目标位置和速度是numpy数组（避免类型错误）
        assert type(target_angles)==np.ndarray
        assert type(target_speeds)==np.ndarray

        # 获取当前关节状态（位置和速度，均为控制自由度长度的数组）
        curr_angles = self.get_act_joint_positions()
        curr_speeds = self.get_act_joint_velocities()

        # 计算误差（目标 - 当前）
        perror = target_angles - curr_angles  # 位置误差
        verror = target_speeds - curr_speeds  # 速度误差

        # 断言：确保增益与误差的维度匹配（防止计算错误）
        assert self.kp.size==perror.size
        assert self.kv.size==verror.size

        # PD控制公式：力矩 = 比例项（kp×位置误差） + 微分项（kv×速度误差）
        return self.kp * perror + self.kv * verror

    def set_motor_torque(self, torque, motor_dyn_fwd = False):
        """
        Apply torques to motors.
        """
        """
        将计算得到的力矩应用到电机（执行器）上，可选启用电机动态模型修正。

        参数：
            torque (np.ndarray/list)：待应用的力矩数组，长度必须等于控制自由度。
            motor_dyn_fwd (bool)：是否启用电机动态模型（默认False）。若为True，需先加载电机模型（load_motor_nets）。

        功能：
            - 验证输入力矩的合法性（类型和维度）。
            - 若启用电机动态模型，调用motor_nets_forward修正力矩（模拟真实电机特性）。
            - 将最终力矩写入MuJoCo的执行器控制信号（self.data.ctrl）。
        """
        # 处理输入：确保力矩是numpy数组，且维度正确
        if isinstance(torque, np.ndarray):
            assert torque.shape==(self.nu(), )  # 验证长度等于控制自由度
            ctrl = torque
        elif isinstance(torque, list):
            assert len(torque)==self.nu()       # 列表长度验证
            ctrl = np.copy(torque)              # 转换为numpy数组
        else:
            raise Exception("motor torque should be list of ndarray.") # 非法类型报错
        try:
            # 若启用电机动态模型，修正力矩（模拟真实电机的延迟、摩擦等特性）
            if motor_dyn_fwd:
                # 检查电机模型是否已加载
                if not hasattr(self, 'motor_dyn_nets'):
                    raise Exception("motor dynamics network are not defined.")
                # 获取齿轮比（电机到关节的传动比）
                gear = self.get_gear_ratios()
                # 调用电机模型修正力矩（先转换到电机层面，修正后再转换回关节层面）
                ctrl = self.motor_nets_forward(ctrl*gear) # 乘以齿轮比：关节力矩→电机力矩
                ctrl /= gear # 除以齿轮比：电机力矩→关节力矩
                # 将修正后的控制信号写入MuJoCo的执行器（ctrl是MuJoCo中执行器的控制接口）
            np.copyto(self.data.ctrl, ctrl)
        except Exception as e:
            print("Could not apply motor torque.")
            print(e)
        return

    
    def step(self, mj_step=True, nstep=1):
        """
        推进物理仿真，更新与位置和速度相关的物理场（如碰撞、力、加速度）。

        参数：
            mj_step (bool)：是否使用MuJoCo的完整仿真函数（mj_step）。
                - True：直接调用mj_step推进nstep步（简单模式）。
                - False：分步骤调用mj_step1/mj_step2（灵活模式，适应自定义积分）。
            nstep (int)：需要推进的仿真步数（默认1步）。每步对应模型的timestep（如0.002s）。

        改编自：dm_control/mujoco/engine.py，适配自定义仿真控制需求。
        """
        # 模式1：使用MuJoCo的完整仿真函数mj_step
        if mj_step:
            # mj_step：MuJoCo的核心仿真函数，执行完整的物理步骤：
            # 包括碰撞检测、力计算、积分更新（位置/速度）等
            mujoco.mj_step(self.model, self.data, nstep)
            # 包括碰撞检测、力计算、积分更新（位置/速度）等
            self.stepCounter += nstep
            return

        # 模式2：分步骤调用仿真子函数（适用于自定义积分或特殊需求）
        # 处理不同积分器（MuJoCo支持欧拉积分和RK4积分）
        # 欧拉积分（mjINT_EULER）：一阶积分，速度快但精度低
        # RK4积分（mjINT_RK4）：四阶龙格-库塔，精度高但计算量大
        if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4.value:
          # 对于欧拉积分：假设已调用mj_step1（位置更新），此处执行mj_step2（速度/力更新）
          mujoco.mj_step2(self.model, self.data)
          # 若步数>1，剩余步数用mj_step处理（避免重复写分步骤逻辑）
          if nstep > 1:
            mujoco.mj_step(self.model, self.data, nstep-1)
        else:
          # 对于RK4积分：直接用mj_step处理所有步数（RK4不适合分步骤调用）
          mujoco.mj_step(self.model, self.data, nstep)

        # 无论哪种积分器，最后调用mj_step1更新位置相关场（确保状态同步）
        # mj_step1：处理位置约束、更新qpos相关的派生量（如xmat、xpos）
        mujoco.mj_step1(self.model, self.data)

        # 更新步数计数器
        self.stepCounter += nstep
