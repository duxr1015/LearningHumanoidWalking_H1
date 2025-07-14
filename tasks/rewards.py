import numpy as np

##############################
##############################
# 此处定义奖励函数
##############################
##############################

def _calc_fwd_vel_reward(self):
    """
    计算前进速度奖励（鼓励机器人按目标速度前进）
    返回：基于速度误差的奖励值（指数衰减）
    """
    # 获取根节点（如骨盆）的线速度（x方向，即前进方向）
    root_vel = self._client.get_qvel()[0]
    # 计算实际速度与目标速度的误差（L2范数）
    error = np.linalg.norm(root_vel - self._goal_speed_ref)
    # 用指数函数将误差转换为奖励（误差越小，奖励越接近1）
    return np.exp(-error)

def _calc_action_reward(self, action, prev_action):
    """
    计算动作平滑性奖励（惩罚动作突变，鼓励平滑控制）
    参数：
        action：当前动作
        prev_action：上一步动作
    返回：基于动作变化量的奖励值（指数衰减）
    """
    # 计算动作变化的平均绝对值（作为惩罚项）
    penalty = 5 * sum(np.abs(prev_action - action)) / len(action)
    # 用指数函数将惩罚转换为奖励（动作变化越小，奖励越接近1）
    return np.exp(-penalty)

def _calc_torque_reward(self, prev_torque):
    """
    计算关节力矩奖励（惩罚力矩突变，鼓励平滑用力）
    参数：
        prev_torque：上一步的关节力矩
    返回：基于力矩变化量的奖励值（指数衰减）
    """
    # 获取当前关节力矩并转换为数组
    torque = np.asarray(self._client.get_act_joint_torques())
    # 计算力矩变化的平均绝对值（作为惩罚项）
    penalty = 0.25 * (sum(np.abs(prev_torque - torque)) / len(torque))
    # 用指数函数将惩罚转换为奖励（力矩变化越小，奖励越接近1）
    return np.exp(-penalty)

def _calc_height_reward(self):
    """
    计算高度奖励（鼓励机器人保持目标高度）
    返回：基于高度误差的奖励值（指数衰减）
    """
    # 计算脚部与地面的接触点高度（取最低接触点）
    if self._client.check_rfoot_floor_collision() or self._client.check_lfoot_floor_collision():
        # 收集左右脚与地面的所有接触点，取z坐标最小值
        contact_point = min([c.pos[2] for _, c in (self._client.get_rfoot_floor_contacts() +
                                                  self._client.get_lfoot_floor_contacts())])
    else:
        # 无接触时默认接触点为0
        contact_point = 0
    # 获取根节点（如骨盆）的z坐标（当前高度）
    current_height = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[2]
    # 计算相对高度（根节点高度 - 接触点高度）
    relative_height = current_height - contact_point
    # 计算相对高度与目标高度的误差
    error = np.abs(relative_height - self._goal_height_ref)
    # 定义死区（误差在此范围内不惩罚，速度越快死区越大）
    deadzone_size = 0.01 + 0.05 * self._goal_speed_ref
    if error < deadzone_size:
        error = 0
    # 用指数函数将误差转换为奖励（高度越稳定，奖励越接近1）
    return np.exp(-40 * np.square(error))

def _calc_heading_reward(self):
    """
    计算航向奖励（鼓励机器人沿目标方向前进）
    返回：基于航向误差的奖励值（指数衰减）
    """
    # 获取根节点的线速度（前3个元素为x,y,z方向速度）
    cur_heading = self._client.get_qvel()[:3]
    # 归一化航向向量
    cur_heading /= np.linalg.norm(cur_heading)
    # 计算当前航向与目标航向（x轴正方向）的误差（L2范数）
    error = np.linalg.norm(cur_heading - np.array([1, 0, 0]))
    # 用指数函数将误差转换为奖励（航向越准，奖励越接近1）
    return np.exp(-error)

def _calc_root_accel_reward(self):
    """
    计算根节点加速度奖励（惩罚过大的旋转和线加速度，鼓励平稳运动）
    返回：基于加速度的奖励值（指数衰减）
    """
    # 获取关节速度（前6个元素为根节点的线速度和角速度）
    qvel = self._client.get_qvel()
    # 获取关节加速度（前3个元素为根节点的线加速度）
    qacc = self._client.get_qacc()
    # 计算惩罚项：旋转速度（qvel[3:6]）和线加速度（qacc[0:3]）的绝对值之和
    error = 0.25 * (np.abs(qvel[3:6]).sum() + np.abs(qacc[0:3]).sum())
    # 用指数函数将惩罚转换为奖励（运动越平稳，奖励越接近1）
    return np.exp(-error)

def _calc_feet_separation_reward(self):
    """
    计算双脚分离度奖励（鼓励双脚保持合适的横向距离）
    返回：基于双脚距离误差的奖励值（指数衰减）
    """
    # 获取左右脚在y方向（横向）的位置
    rfoot_pos = self._client.get_rfoot_body_pos()[1]
    lfoot_pos = self._client.get_lfoot_body_pos()[1]
    # 计算双脚横向距离的绝对值
    foot_dist = np.abs(rfoot_pos - lfoot_pos)
    # 计算距离与目标值（0.35米）的平方误差（放大5倍）
    error = 5 * np.square(foot_dist - 0.35)
    # 死区设置：距离在0.3-0.4米范围内无惩罚
    if foot_dist < 0.40 and foot_dist > 0.30:
        error = 0
    # 用指数函数将误差转换为奖励（距离越合适，奖励越接近1）
    return np.exp(-error)

def _calc_foot_frc_clock_reward(self, left_frc_fn, right_frc_fn):
    """
    基于步态相位的脚部力奖励（根据步态周期约束脚部受力）
    参数：
        left_frc_fn：左脚力的相位函数
        right_frc_fn：右脚力的相位函数
    返回：左右脚力的综合奖励
    """
    # 计算单脚最大期望地面反作用力（体重的一半）
    desired_max_foot_frc = self._client.get_robot_mass() * 9.8 * 0.5
    # 归一化左右脚实际受力（限制最大值，映射到[0,1]）
    normed_left_frc = min(self.l_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(self.r_foot_frc, desired_max_foot_frc) / desired_max_foot_frc
    # 将归一化范围从[0,1]映射到[-1,1]
    normed_left_frc *= 2
    normed_left_frc -= 1
    normed_right_frc *= 2
    normed_right_frc -= 1

    # 根据当前相位获取左右脚力的期望相位值
    left_frc_clock = left_frc_fn(self._phase)
    right_frc_clock = right_frc_fn(self._phase)

    # 计算左右脚力的得分（通过tan函数放大符合期望的行为）
    left_frc_score = np.tan(np.pi / 4 * left_frc_clock * normed_left_frc)
    right_frc_score = np.tan(np.pi / 4 * right_frc_clock * normed_right_frc)

    # 综合左右脚得分（取平均）
    foot_frc_score = (left_frc_score + right_frc_score) / 2
    return foot_frc_score

def _calc_foot_vel_clock_reward(self, left_vel_fn, right_vel_fn):
    """
    基于步态相位的脚部速度奖励（根据步态周期约束脚部速度）
    参数：
        left_vel_fn：左脚速度的相位函数
        right_vel_fn：右脚速度的相位函数
    返回：左右脚速度的综合奖励
    """
    # 设定单脚最大期望速度
    desired_max_foot_vel = 0.2
    # 归一化左右脚实际速度（限制最大值，映射到[0,1]）
    normed_left_vel = min(np.linalg.norm(self.l_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(self.r_foot_vel), desired_max_foot_vel) / desired_max_foot_vel
    # 将归一化范围从[0,1]映射到[-1,1]
    normed_left_vel *= 2
    normed_left_vel -= 1
    normed_right_vel *= 2
    normed_right_vel -= 1

    # 根据当前相位获取左右脚速度的期望相位值
    left_vel_clock = left_vel_fn(self._phase)
    right_vel_clock = right_vel_fn(self._phase)

    # 计算左右脚速度的得分（通过tan函数放大符合期望的行为）
    left_vel_score = np.tan(np.pi / 4 * left_vel_clock * normed_left_vel)
    right_vel_score = np.tan(np.pi / 4 * right_vel_clock * normed_right_vel)

    # 综合左右脚得分（取平均）
    foot_vel_score = (left_vel_score + right_vel_score) / 2
    return foot_vel_score

def _calc_foot_pos_clock_reward(self):
    """
    基于步态相位的脚部位置奖励（根据步态周期约束脚部高度）
    返回：左右脚高度的综合奖励
    """
    # 设定最大期望脚部高度
    desired_max_foot_height = 0.05
    # 获取左右脚力传感器的z坐标（脚部高度）
    l_foot_pos = self._client.get_object_xpos_by_name('lf_force', 'OBJ_SITE')[2]
    r_foot_pos = self._client.get_object_xpos_by_name('rf_force', 'OBJ_SITE')[2]
    # 归一化左右脚高度（限制最大值，映射到[0,1]）
    normed_left_pos = min(np.linalg.norm(l_foot_pos), desired_max_foot_height) / desired_max_foot_height
    normed_right_pos = min(np.linalg.norm(r_foot_pos), desired_max_foot_height) / desired_max_foot_height

    # 根据当前相位获取左右脚位置的期望相位值
    left_pos_clock = self.left_clock[1](self._phase)
    right_pos_clock = self.right_clock[1](self._phase)

    # 计算左右脚位置的得分（通过tan函数放大符合期望的行为）
    left_pos_score = np.tan(np.pi / 4 * left_pos_clock * normed_left_pos)
    right_pos_score = np.tan(np.pi / 4 * right_pos_clock * normed_right_pos)

    # 综合左右脚得分（求和）
    foot_pos_score = left_pos_score + right_pos_score
    return foot_pos_score

def _calc_body_orient_reward(self, body_name, quat_ref=[1, 0, 0, 0]):
    """
    计算身体姿态奖励（鼓励身体部位保持目标朝向）
    参数：
        body_name：身体部位名称
        quat_ref：目标姿态的四元数（默认[1,0,0,0]，即无旋转）
    返回：基于姿态误差的奖励值（指数衰减）
    """
    # 获取指定身体部位的当前姿态（四元数）
    body_quat = self._client.get_object_xquat_by_name(body_name, "OBJ_BODY")
    # 目标姿态四元数
    target_quat = np.array(quat_ref)
    # 计算姿态误差（基于四元数内积，误差越小内积越接近1）
    error = 10 * (1 - np.inner(target_quat, body_quat) ** 2)
    # 用指数函数将误差转换为奖励（姿态越接近目标，奖励越接近1）
    return np.exp(-error)

def _calc_joint_vel_reward(self, enabled, cutoff=0.5):
    """
    计算关节速度奖励（惩罚关节超速，鼓励速度在安全范围内）
    参数：
        enabled：启用的关节索引列表
        cutoff：速度阈值比例（默认0.5，即最大速度的50%）
    返回：基于关节超速的奖励值（指数衰减）
    """
    # 获取电机速度和速度限制
    motor_speeds = self._client.get_motor_velocities()
    motor_limits = self._client.get_motor_speed_limits()
    # 筛选启用的关节
    motor_speeds = [motor_speeds[i] for i in enabled]
    motor_limits = [motor_limits[i] for i in enabled]
    # 计算超速关节的速度平方和（作为惩罚项）
    error = 5e-6 * sum([np.square(q)
                      for q, qmax in zip(motor_speeds, motor_limits)
                      if np.abs(q) > np.abs(cutoff * qmax)])
    # 用指数函数将惩罚转换为奖励（超速越小，奖励越接近1）
    return np.exp(-error)


def _calc_joint_acc_reward(self):
    """
    计算关节加速度奖励（惩罚过大的关节加速度）
    返回：基于关节加速度的惩罚值（需结合权重使用）
    """
    # 计算关节加速度的平方和
    joint_acc_cost = np.sum(np.square(self._client.get_qacc()[-self._num_joints:]))
    # 乘以权重返回惩罚值（注意：此函数返回的是惩罚，非奖励）
    return self.wp.joint_acc_weight * joint_acc_cost

def _calc_ang_vel_reward(self):
    """
    计算角速度奖励（惩罚过大的旋转角速度）
    返回：基于旋转角速度的惩罚值（需结合权重使用）
    """
    # 获取根节点的旋转角速度（后3个元素）
    ang_vel = self._client.get_qvel()[3:6]
    # 计算角速度的平方范数
    ang_vel_cost = np.square(np.linalg.norm(ang_vel))
    # 乘以权重返回惩罚值（注意：此函数返回的是惩罚，非奖励）
    return self.wp.ang_vel_weight * ang_vel_cost

def _calc_impact_reward(self):
    """
    计算碰撞冲击奖励（惩罚脚部与地面的剧烈碰撞）
    返回：基于碰撞冲击力的惩罚值（需结合权重使用）
    """
    # 计算左右脚与地面的接触数量（原代码存在拼写错误：contactts→contacts）
    ncon = len(self._client.get_rfoot_floor_contacts()) + \
           len(self._client.get_lfoot_floor_contacts())
    if ncon == 0:
        return 0
    # 计算平均冲击力的平方和
    quad_impact_cost = np.sum(np.square(self._client.get_body_ext_force())) / ncon
    # 乘以权重返回惩罚值（注意：此函数返回的是惩罚，非奖励）
    return self.wp.impact_weight * quad_impact_cost

def _calc_zmp_reward(self):
    """
    计算零力矩点（ZMP）奖励（鼓励ZMP保持在支撑范围内）
    返回：基于ZMP误差的惩罚值（需结合权重使用）
    """
    # 估计当前ZMP
    self.current_zmp = estimate_zmp(self)
    # 过滤异常值（ZMP突变超过1时，沿用之前的值）
    if np.linalg.norm(self.current_zmp - self._prev_zmp) > 1:
        self.current_zmp = self._prev_zmp
    # 计算当前ZMP与期望ZMP的平方误差
    zmp_cost = np.square(np.linalg.norm(self.current_zmp - self.desired_zmp))
    # 更新上一时刻ZMP
    self._prev_zmp = self.current_zmp
    # 乘以权重返回惩罚值（注意：此函数返回的是惩罚，非奖励）
    return self.wp.zmp_weight * zmp_cost

def _calc_foot_contact_reward(self):
    """
    计算脚部接触奖励（惩罚脚部与地面的异常接触位置）
    返回：基于接触位置的惩罚值（需结合权重使用）
    """
    # 获取左右脚与地面的碰撞信息
    right_contacts = self._client.get_rfoot_floor_collisions()
    left_contacts = self._client.get_lfoot_floor_collisions()

    # 设定接触位置半径阈值
    radius_thresh = 0.3
    # 获取根节点的x,y坐标（基准位置）
    f_base = self._client.get_qpos()[0:2]
    # 计算右脚接触点与基准位置的距离（超过阈值的部分）
    c_dist_r = [(np.linalg.norm(c.pos[0:2] - f_base)) for _, c in right_contacts]
    # 计算左脚接触点与基准位置的距离（超过阈值的部分）
    c_dist_l = [(np.linalg.norm(c.pos[0:2] - f_base)) for _, c in left_contacts]
    # 求和超过阈值的距离作为惩罚
    d = sum([r for r in c_dist_r if r > radius_thresh] +
            [r for r in c_dist_l if r > radius_thresh])
    # 乘以权重返回惩罚值（注意：此函数返回的是惩罚，非奖励）
    return self.wp.foot_contact_weight * d

def _calc_gait_reward(self):
    """
    计算步态奖励（根据步态相位约束脚部行为）
    返回：基于步态相位的惩罚值（需结合权重使用）
    """
    if self._period <= 0:
        raise Exception("周期应大于零。")

    # 获取左右脚地面反作用力
    rfoot_grf = self._client.get_rfoot_grf()
    lfoot_grf = self._client.get_lfoot_grf()

    # 获取左右脚速度
    rfoot_speed = self._client.get_rfoot_body_speed()
    lfoot_speed = self._client.get_lfoot_body_speed()

    # 获取左右脚位置
    rfoot_pos = self._client.get_rfoot_body_pos()
    lfoot_pos = self._client.get_lfoot_body_pos()
    # 设定摆动和支撑阶段的目标高度
    swing_height = 0.3
    stance_height = 0.1

    # 相位阈值（0.5为周期中点）
    r = 0.5
    if self._phase < r:
        # 右相位：右脚支撑，左脚摆动（惩罚左脚受力）
        cost = (0.01 * lfoot_grf)  # 原代码注释了高度惩罚项
    else:
        # 左相位：左脚支撑，右脚摆动（惩罚右脚受力）
        cost = (0.01 * rfoot_grf)
    # 乘以权重返回惩罚值（注意：此函数返回的是惩罚，非奖励）
    return self.wp.gait_weight * cost

def _calc_reference(self):
    """
    计算参考轨迹奖励（鼓励关节位置跟踪参考轨迹）
    返回：基于轨迹跟踪误差的惩罚值（需结合权重使用）
    """
    if self.ref_poses is None:
        raise Exception("未提供参考轨迹。")

    # 根据当前相位获取参考轨迹索引
    phase = self._phase
    traj_length = self.traj_len
    indx = int(phase * (traj_length - 1))
    reference_pose = self.ref_poses[indx, :]

    # 获取当前关节位置
    current_pose = np.array(self._client.get_act_joint_positions())

    # 计算轨迹跟踪误差的平方范数
    cost = np.square(np.linalg.norm(reference_pose - current_pose))
    # 乘以权重返回惩罚值（注意：此函数返回的是惩罚，非奖励）
    return self.wp.ref_traj_weight * cost

##############################
##############################
# 定义工具函数
##############################
##############################

def estimate_zmp(self):
    """
    估计零力矩点（ZMP），用于评估机器人平衡状态
    返回：ZMP的x,y坐标
    """
    Gv = 9.80665  # 重力加速度
    Mg = self._mass * Gv  # 机器人总重力

    # 获取质心位置、线动量、角动量（subtree_com[1]通常为整体质心）
    com_pos = self._sim.data.subtree_com[1].copy()
    lin_mom = self._sim.data.subtree_linvel[1].copy() * self._mass  # 线动量 = 质量×线速度
    ang_mom = self._sim.data.subtree_angmom[1].copy() + np.cross(com_pos, lin_mom)  # 角动量 = 自身角动量 + 质心交叉项

    # 计算线动量和角动量的变化率（近似加速度）
    d_lin_mom = (lin_mom - self._prev_lin_mom) / self._control_dt  # 线动量变化率（力）
    d_ang_mom = (ang_mom - self._prev_ang_mom) / self._control_dt  # 角动量变化率（力矩）

    # 计算垂直方向的合力（z方向）
    Fgz = d_lin_mom[2] + Mg

    # 检查与地面的接触
    contacts = [self._sim.data.contact[i] for i in range(self._sim.data.ncon)]
    contact_flag = [(c.geom1 == 0 or c.geom2 == 0) for c in contacts]  # 判断是否与地面（geom0）接触

    # 计算ZMP（零力矩点）
    if (True in contact_flag) and Fgz > 20:  # 有接触且垂直力足够大时
        # ZMP公式：x = (Mg*com_x - d_ang_mom_y)/Fgz；y = (Mg*com_y + d_ang_mom_x)/Fgz
        zmp_x = (Mg * com_pos[0] - d_ang_mom[1]) / Fgz
        zmp_y = (Mg * com_pos[1] + d_ang_mom[0]) / Fgz
    else:
        # 无有效接触时，ZMP近似为质心位置
        zmp_x = com_pos[0]
        zmp_y = com_pos[1]

    # 更新上一时刻的动量
    self._prev_lin_mom = lin_mom
    self._prev_ang_mom = ang_mom
    return np.array([zmp_x, zmp_y])

##############################
##############################
# 基于apex的相位奖励生成
##############################
##############################

def create_phase_reward(swing_duration, stance_duration, strict_relaxer, stance_mode, FREQ=40):
    """
    创建基于步态相位的奖励函数（生成随相位变化的期望行为曲线）
    参数：
        swing_duration：摆动阶段持续时间（秒）
        stance_duration：支撑阶段持续时间（秒）
        strict_relaxer：相位过渡的平滑程度（0-1，值越小过渡越平滑）
        stance_mode：支撑模式（"grounded"接地/"aerial"空中/"zero"零约束）
        FREQ：控制频率（Hz，默认40）
    返回：
        右脚和左脚的相位函数（力和速度各一个）
    """
    from scipy.interpolate import PchipInterpolator  # 导入分段三次 hermite 插值函数

    # 将时间（秒）转换为相位步数（= 时间×频率）
    right_swing = np.array([0.0, swing_duration]) * FREQ  # 右脚摆动阶段
    first_dblstance = np.array([swing_duration, swing_duration + stance_duration]) * FREQ  # 第一次双足支撑阶段
    left_swing = np.array([swing_duration + stance_duration, 2 * swing_duration + stance_duration]) * FREQ  # 左脚摆动阶段
    second_dblstance = np.array([2 * swing_duration + stance_duration, 2 * (swing_duration + stance_duration)]) * FREQ  # 第二次双足支撑阶段

    # 初始化相位点（2行8列：行0为相位步，行1为期望输出值）
    r_frc_phase_points = np.zeros((2, 8))  # 右脚力相位点
    r_vel_phase_points = np.zeros((2, 8))  # 右脚速度相位点
    l_frc_phase_points = np.zeros((2, 8))  # 左脚力相位点
    l_vel_phase_points = np.zeros((2, 8))  # 左脚速度相位点

    # 右脚摆动阶段：设置相位过渡的松弛偏移（平滑过渡区域）
    right_swing_relax_offset = (right_swing[1] - right_swing[0]) * strict_relaxer
    # 相位点0-1：右脚摆动阶段的过渡区间
    l_frc_phase_points[0,0] = r_frc_phase_points[0,0] = right_swing[0] + right_swing_relax_offset
    l_frc_phase_points[0,1] = r_frc_phase_points[0,1] = right_swing[1] - right_swing_relax_offset
    l_vel_phase_points[0,0] = r_vel_phase_points[0,0] = right_swing[0] + right_swing_relax_offset
    l_vel_phase_points[0,1] = r_vel_phase_points[0,1] = right_swing[1] - right_swing_relax_offset
    # 右脚摆动阶段的期望行为：左脚用力、右脚移动（惩罚左脚速度和右脚力，奖励左脚力和右脚速度）
    l_vel_phase_points[1,:2] = r_frc_phase_points[1,:2] = np.negative(np.ones(2))  # 惩罚项（-1）
    l_frc_phase_points[1,:2] = r_vel_phase_points[1,:2] = np.ones(2)  # 奖励项（1）

    # 第一次双足支撑阶段：设置松弛偏移
    dbl_stance_relax_offset = (first_dblstance[1] - first_dblstance[0]) * strict_relaxer
    # 相位点2-3：第一次双足支撑阶段的过渡区间
    l_frc_phase_points[0,2] = r_frc_phase_points[0,2] = first_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0,3] = r_frc_phase_points[0,3] = first_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0,2] = r_vel_phase_points[0,2] = first_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0,3] = r_vel_phase_points[0,3] = first_dblstance[1] - dbl_stance_relax_offset
    # 双足支撑阶段的期望行为（根据模式调整）
    if stance_mode == "aerial":
        # 空中模式：奖励速度，惩罚力
        l_frc_phase_points[1,2:4] = r_frc_phase_points[1,2:4] = np.negative(np.ones(2))
        l_vel_phase_points[1,2:4] = r_vel_phase_points[1,2:4] = np.ones(2)
    elif stance_mode == "zero":
        # 零约束模式：无奖励/惩罚
        l_frc_phase_points[1,2:4] = r_frc_phase_points[1,2:4] = np.zeros(2)
        l_vel_phase_points[1,2:4] = r_vel_phase_points[1,2:4] = np.zeros(2)
    else:
        # 接地模式：奖励力，惩罚速度
        l_frc_phase_points[1,2:4] = r_frc_phase_points[1,2:4] = np.ones(2)
        l_vel_phase_points[1,2:4] = r_vel_phase_points[1,2:4] = np.negative(np.ones(2))

    # 左脚摆动阶段：设置松弛偏移
    left_swing_relax_offset = (left_swing[1] - left_swing[0]) * strict_relaxer
    # 相位点4-5：左脚摆动阶段的过渡区间
    l_frc_phase_points[0,4] = r_frc_phase_points[0,4] = left_swing[0] + left_swing_relax_offset
    l_frc_phase_points[0,5] = r_frc_phase_points[0,5] = left_swing[1] - left_swing_relax_offset
    l_vel_phase_points[0,4] = r_vel_phase_points[0,4] = left_swing[0] + left_swing_relax_offset
    l_vel_phase_points[0,5] = r_vel_phase_points[0,5] = left_swing[1] - left_swing_relax_offset
    # 左脚摆动阶段的期望行为：右脚用力、左脚移动（惩罚右脚速度和左脚力，奖励右脚力和左脚速度）
    l_vel_phase_points[1,4:6] = r_frc_phase_points[1,4:6] = np.ones(2)  # 奖励项
    l_frc_phase_points[1,4:6] = r_vel_phase_points[1,4:6] = np.negative(np.ones(2))  # 惩罚项

    # 第二次双足支撑阶段：设置松弛偏移
    dbl_stance_relax_offset = (second_dblstance[1] - second_dblstance[0]) * strict_relaxer
    # 相位点6-7：第二次双足支撑阶段的过渡区间
    l_frc_phase_points[0,6] = r_frc_phase_points[0,6] = second_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0,7] = r_frc_phase_points[0,7] = second_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0,6] = r_vel_phase_points[0,6] = second_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0,7] = r_vel_phase_points[0,7] = second_dblstance[1] - dbl_stance_relax_offset
    # 第二次双足支撑阶段的期望行为（同第一次）
    if stance_mode == "aerial":
        l_frc_phase_points[1,6:] = r_frc_phase_points[1,6:] = np.negative(np.ones(2))
        l_vel_phase_points[1,6:] = r_vel_phase_points[1,6:] = np.ones(2)
    elif stance_mode == "zero":
        l_frc_phase_points[1,6:] = r_frc_phase_points[1,6:] = np.zeros(2)
        l_vel_phase_points[1,6:] = r_vel_phase_points[1,6:] = np.zeros(2)
    else:
        l_frc_phase_points[1,6:] = r_frc_phase_points[1,6:] = np.ones(2)
        l_vel_phase_points[1,6:] = r_vel_phase_points[1,6:] = np.negative(np.ones(2))

    ## 扩展数据到三个周期（前一个、当前、后一个），确保相位连续性
    # 前一个周期的相位点（偏移当前周期）
    r_frc_prev_cycle = np.copy(r_frc_phase_points)
    r_vel_prev_cycle = np.copy(r_vel_phase_points)
    l_frc_prev_cycle = np.copy(l_frc_phase_points)
    l_vel_prev_cycle = np.copy(l_vel_phase_points)
    l_frc_prev_cycle[0] = r_frc_prev_cycle[0] = r_frc_phase_points[0] - r_frc_phase_points[0, -1] - dbl_stance_relax_offset
    l_vel_prev_cycle[0] = r_vel_prev_cycle[0] = r_vel_phase_points[0] - r_vel_phase_points[0, -1] - dbl_stance_relax_offset

    # 后一个周期的相位点（偏移当前周期）
    r_frc_second_cycle = np.copy(r_frc_phase_points)
    r_vel_second_cycle = np.copy(r_vel_phase_points)
    l_frc_second_cycle = np.copy(l_frc_phase_points)
    l_vel_second_cycle = np.copy(l_vel_phase_points)
    l_frc_second_cycle[0] = r_frc_second_cycle[0] = r_frc_phase_points[0] + r_frc_phase_points[0, -1] + dbl_stance_relax_offset
    l_vel_second_cycle[0] = r_vel_second_cycle[0] = r_vel_phase_points[0] + r_vel_phase_points[0, -1] + dbl_stance_relax_offset

    # 合并三个周期的相位点（确保插值连续性）
    r_frc_phase_points_repeated = np.hstack((r_frc_prev_cycle, r_frc_phase_points, r_frc_second_cycle))
    r_vel_phase_points_repeated = np.hstack((r_vel_prev_cycle, r_vel_phase_points, r_vel_second_cycle))
    l_frc_phase_points_repeated = np.hstack((l_frc_prev_cycle, l_frc_phase_points, l_frc_second_cycle))
    l_vel_phase_points_repeated = np.hstack((l_vel_prev_cycle, l_vel_phase_points, l_vel_second_cycle))

    ## 用三次样条插值创建平滑的相位函数（并限制输出在[-1,1]）
    r_frc_phase_spline = PchipInterpolator(r_frc_phase_points_repeated[0], r_frc_phase_points_repeated[1])
    r_vel_phase_spline = PchipInterpolator(r_vel_phase_points_repeated[0], r_vel_phase_points_repeated[1])
    l_frc_phase_spline = PchipInterpolator(l_frc_phase_points_repeated[0], l_frc_phase_points_repeated[1])
    l_vel_phase_spline = PchipInterpolator(l_vel_phase_points_repeated[0], l_vel_phase_points_repeated[1])

    # 返回右脚和左脚的相位函数（力和速度）
    return [r_frc_phase_spline, r_vel_phase_spline], [l_frc_phase_spline, l_vel_phase_spline]