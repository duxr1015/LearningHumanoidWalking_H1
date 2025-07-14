import numpy as np

class RobotBase(object):
    def __init__(self, pdgains, dt, client, task, pdrand_k=0,sim_bemf = False, sim_motor_dyn = False):

        self.client = client  # �����˽ӿ�
        self.task = task      # �����߼�
        self.control_dt = dt  # ����ʱ�䲽��
        self.pdrand_k = pdrand_k # PD���������ϵ��,PD ���������ϵ������ 0.1 ��ʾ ��10% �������������ǿ����³����
        self.sim_bemf = sim_bemf # �Ƿ�ģ�ⷴ�綯�ƣ���ؽ��ٶȳ����ȵ�������������������ʵ��
        self.sim_motor_dyn = sim_motor_dyn  #�Ƿ�ģ��������ѧ,ģ������̬��Ӧ�����ӳ١����ͣ�������ȷ�ؽ�ģ�����Ϊ

        #��ֹͬʱģ�ⷴ�綯�ƺ͵������ѧ(�����ͻ)
        assert(self.sim_bemf & self.sim_motor_dyn == False), \
            "You cannot simulate back-EMF and motor dynamics simultaneously!"
        
        # ����PD���棨���ؽ����ã�
        self.kp = pdgains[0]  # ��������
        self.kd = pdgains[1]  # ΢������
        assert self.kp.shape==self.kd.shape==(self.client.nu(),), \
            f"kp shape {self.kp.shape} and kd shape {self.kd.shape} must be {(self.client.nu(),)}"
        
        # ���綯������ϵ��������ģ����������
        self.tau_d = np.zeros(self.client.nu())

        # ��ʼ��PD����������֤
        self.client.set_pd_gains(self.kp, self.kd)
        tau = self.client.step_pd(np.zeros(self.client.nu()), np.zeros(self.client.nu()))
        w = self.client.get_act_joint_velocities()
        assert len(w)==len(tau)

        # ��¼��ʷ���������أ����ڽ������㣩
        self.prev_action = None
        self.prev_torque = None
        self.iteration_count = np.inf

        # ����֡�������������Ʋ���/���沽����
        if (np.around(self.control_dt%self.client.sim_dt(), 6)):
            raise Exception("Control dt should be an integer multiple of Simulation dt.")
        self.frame_skip = int(self.control_dt/self.client.sim_dt())

    def _do_simulation(self, target, n_frames):
        # �����PD���棨��ǿ³���ԣ�
        if self.pdrand_k:
            k = self.pdrand_k
            kp = np.random.uniform((1-k)*self.kp, (1+k)*self.kp)
            kd = np.random.uniform((1-k)*self.kd, (1+k)*self.kd)
            self.client.set_pd_gains(kp, kd)

        assert target.shape == (self.client.nu(),), \
            f"Target shape must be {(self.client.nu(),)}"

        ratio = self.client.get_gear_rations()  # ��ȡ���ֱȣ�ת���������Ϊ�ؽ����أ�

        # ������·��綯������ϵ��������sim_bemf=True��10%����ʱ��
        if self.sim_bemf and np.random.randint(10)==0:
            self.tau_d = np.random.uniform(5, 40, self.client.nu())

        # ִ�ж�η��沽��frame_skip�Σ�
        for _ in range(n_frames):
            w = self.client.get_act_joint_velocities()  # ��ȡ��ǰ�ؽ��ٶ�
            tau = self.client.step_pd(target, np.zeros(self.client.nu()))  # PD��������������

            # Ӧ�÷��綯�ƣ��ٶ���ص�������
            if self.sim_bemf:
                tau = tau - self.tau_d*w

            # ���ǳ��ֱȣ�������ء��ؽ����ص�ת����
            tau /= ratio

            # ���õ�����ز��ƽ�����
            self.client.set_motor_torque(tau, self.sim_motor_dyn)
            self.client.step()

    def step(self, action, offset=None):
        # ���ͺ�ά�ȼ��
        if not isinstance(action, np.ndarray):
            raise TypeError("Expected action to be a numpy array")
        action = np.copy(action)
        assert action.shape == (self.client.nu(),), \
            f"Action vector length expected to be: {self.client.nu()} but is {action.shape}"
        
        # Ӧ��λ��ƫ�ƣ������������̬�����λ�ã�
        if offset is not None:
            if not isinstance(offset, np.ndarray):
                raise TypeError("Expected offset to be a numpy array")
            assert offset.shape == action.shape, \
                f"Offset shape {offset} must match action shape {action.shape}"
            action += offset
        
        # ��¼��ʷ���������أ����ڽ������㣩
        if self.prev_action is None:
            self.prev_action = action
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.client.get_act_joint_torques())
        
        # ִ�з���
        self._do_simulation(action, self.frame_skip)
        
        # ������ز��������㽱���������ֹ������
        self.task.step()
        rewards = self.task.calc_reward(self.prev_torque, self.prev_action, action)
        done = self.task.done()
        
        # ������ʷ��¼
        self.prev_action = action
        self.prev_torque = np.asarray(self.client.get_act_joint_torques())
        
        return rewards, done

    

    

