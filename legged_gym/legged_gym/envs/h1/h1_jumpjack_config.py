from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1JumpJackCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.05] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_joint' : -0.4,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_joint' : -0.4,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }
        
        randomize_upperbody = False
    
    class domain_rand ( LeggedRobotCfg.domain_rand ):
        
        randomize_dof_bias = False
        max_dof_bias = 0.08
        
        randomize_yaw = True
        init_yaw_range = [-0.5,0.5]

        randomize_ctrl_delay = False
        ctrl_delay_step_range = [0, 1] # integer max real delay is 90ms

        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 0.25
        
        randomize_friction = True
        friction_range = [-0.6, 1.2]
        
        randomize_base_com = False
        class base_com_range: #kg
            x = [-0.06, 0.06]
            y = [-0.1, 0.1]
            z = [-0.15, 0.15]

        randomize_link_mass = False
        link_mass_range = [0.9, 1.2] # *factor
        randomize_link_body_names = [
            'pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 
            'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link',  'torso_link',
        ]

        randomize_pd_gain = False
        kp_range = [0.75, 1.25]
        kd_range = [0.75, 1.25]

        randomize_torque_rfi = False
        rfi_lim = 0.1
        randomize_rfi_lim = False
        rfi_lim_range = [0.5, 1.5]

        randomize_base_mass = False # replaced by randomize_link_mass
        # added_mass_range = [-5., 10.]
        

    class noise ( LeggedRobotCfg.noise ):
        add_noise = False
        noise_level = 1.0 # scales other values
        class noise_scales:
            base_z = 0.05
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            in_contact = 0.1
            height_measurements = 0.05
            last_action = 0.0
            box_3dpos = 0.02
            contact_vel = 0.0
                
        class drift:
            x = 0.03
            y = 0.05
            z = 0.01
            
    class terrain ( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        # mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        curriculum = False
        margin = 10 # [m]
        # curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        terrain_length = 8.
        terrain_width = 8
        num_rows = 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_noise = 187
        num_observations = 187
        
        num_actions = 19

        send_timeouts = False  # full episode rew
        episode_length_s = 10
        period_contact = [0.4, 0.8]
      
    class commands ( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 0.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [.0, .0] # min max [m/s]
            lin_vel_y = [.0, .0]   # min max [m/s]
            ang_vel_yaw = [.0, .0]    # min max [rad/s]
            heading = [.0, .0]
            
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 100,
                     "elbow":100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 6,
                     'ankle': 2,
                     'torso': 6,
                     'shoulder': 2,
                     "elbow":2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

        action_filt = True
        # action_filt = True
        action_cutfreq = 4.0
        torque_effort_scale = 1.0

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        name = "h1"
        foot_name = "ankle"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["pelvis", 
                                       "shoulder", 
                                    #    "elbow",
                                       "hip_roll", 
                                       "hip_pitch", 
                                       "knee",
                                       "torso",
                                       ]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
        
        terminate_by_knee_distance = True
        terminate_by_feet_distance = True
        terminate_by_low_height = True
        terminate_by_lin_vel = True
        terminate_by_ang_vel = False
        terminate_by_gravity = True
        terminate_by_xy = True
        terminate_by_hip_yaw = True
        
        class termination_scales():
            base_height = 0.1
            base_vel = 10.0
            base_ang_vel = 5.0
            gravity_x = 0.7
            gravity_y = 0.7
            min_knee_distance = 0.17
            min_feet_distance = 0.17
            global_xy = 0.8
            hip_yaw_sum = 0.8

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            on_box = 10.
            prev_fulfill = 5.
            
            # hand_x = 1.0
            hand_y = 5.0
            # hand_z = -1.0
            foot_goal = 10.0

            curiosity = 5000.

            # some are not necessary at all
            vel_x = -5. 
            # yaw = -10.0
            rot_z = -0.1
            torques = -0.5
            torque_limits = -500
            dof_acc = -5e-6
            dof_vel = -0.003
            dof_vel_limits = -20.0
            action_rate = -0.1
            dof_pos_limits = -10.0
            termination = -200
            feet_contact_forces = -0.005
            feet_ori_contact = -50.0
            feet_air_time = 10.0
            stumble = -100.0
            contact_velo = -10.
            hand_vel_w = -0.1

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.85 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.8
        soft_torque_limit = 0.8
        max_contact_force = 550. # [N]

        class curiosity:
            obs_dim = 29
            hidden_sizes_hash = [32]
            pred_dim = 16
            obs_lb = [-0.5,-0.5,0.7] + [-0.5,-0.5,-0.5,0.7] + [-0.5,-0.5,-0.5,-1.0,-1.0,-1.0] + [-0.5,-0.5,0.1]*2 + [-0.5,-1.,0.5]*2 + [0.]*4
            obs_ub = [0.5 , 0.5,1.2] + [0.5, 0.5, 0.5, 1.0] + [ 0.5, 0.5, 0.5, 1.0, 1.0, 1.0] + [0.5, 0.5,0.25]*2 + [ 0.8, 1.,2.0]*2 + [1.]*4
            
    class task:
        x_range = 0.15
        y_side = 0.15
        y_stance = 0.3
        y_tol = 0.1
        num_seq = 10
    class normalization:
        class obs_scales: # no normalization for nows
            lin_vel = 1.0 # 2.0
            ang_vel = 1.0 # 0.25
            dof_pos = 1.0 # 1.0
            dof_vel = 1.0 # 0.05
            height_measurements = 1.0 # 5.0
            body_pos = 1.0
            body_lin_vel = 1.0
            body_rot = 1.0
            delta_base_pos = 1.0
            delta_heading = 1.0
        clip_observations = 100.
        clip_actions = 100.
        
class H1JumpJackCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        class_name = "PPO_SYM"
        entropy_coef = 0.001
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'h1:jjack'
        max_iterations = 100000
        
    class policy ( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
  
