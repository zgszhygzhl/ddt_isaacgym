import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F 
from modules.common_modules import MAE, VQVAE, VQVAE_CNN, VQVAE_EMA, VQVAE_RNN, AutoEncoder, BetaVAE, CnnHistoryEncoder, MixedLayerNormMlp, MixedLipMlp, MixedMlp, RnnBarlowTwinsStateHistoryEncoder, RnnDoubleHeadEncoder, RnnEncoder, RnnStateHistoryEncoder, StateHistoryEncoder, VQVAE_Trans, VQVAE_vel, VQVAE_vel_conv, get_activation, mlp_batchnorm_factory, mlp_factory, mlp_layernorm_factory
from modules.normalizer import EmpiricalNormalization
from modules.transformer_modules import ActionCausalTransformer, StateCausalClsTransformer, StateCausalHeadlessTransformer, StateCausalTransformer
class Config:
    def __init__(self):
        self.n_obs = 45
        self.block_size = 9
        self.n_action = 45+3
        self.n_layer: int = 4
        self.n_head: int = 4
        self.n_embd: int = 32
        self.dropout: float = 0.0
        self.bias: bool = True

class CnnActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_actions,
                 priv_encoder_output_dim,
                 actor_hidden_dims=[256, 256, 256],
                 activation='elu'):
        super(CnnActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.priv_encoder_output_dim = priv_encoder_output_dim
        self.activation = activation
        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, priv_encoder_output_dim)
        self.actor_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim,num_actions,actor_hidden_dims,last_act=False)
        self.actor = nn.Sequential(*self.actor_layers)
    
    def forward(self,obs,hist):
        latent = self.history_encoder(hist)
        backbone_input = torch.cat([obs,latent], dim=1)
        mean = self.actor(backbone_input)
        return mean
    
class MlpBarlowTwinsActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBarlowTwinsActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist

        self.obs_normalizer = EmpiricalNormalization(shape=num_prop)
        
        self.mlp_encoder = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=num_prop*num_hist,
                                 out_dims=None,
                                 hidden_dims=mlp_encoder_dims))
        # self.cnn_encoder = CnnHistoryEncoder(num_prop,10,latent_dim)
        
        self.latent_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.vel_layer = nn.Linear(mlp_encoder_dims[-1],3)

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 3,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))

        # self.actor = MixedMlp(input_size=num_prop,
        #                       latent_size=latent_dim+3,
        #                       hidden_size=128,
        #                       num_actions=num_actions,
        #                       num_experts=4)
        
        # self.vel_layer = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
        #                          input_dims=64,
        #                          out_dims=3,
        #                          hidden_dims=[32]))
        
        # self.obs_encoder = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
        #                          input_dims=num_prop,
        #                          out_dims=latent_dim,
        #                          hidden_dims=[64]))
        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        # self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist*2, 3,final_act=False)
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def normalize(self,obs,obs_hist):
        obs = self.obs_normalizer(obs)
        obs_hist = self.obs_normalizer(obs_hist.reshape(-1,self.num_prop)).reshape(-1,10,self.num_prop)
        return obs,obs_hist

    def forward(self,obs,obs_hist):
        obs,obs_hist = self.normalize(obs,obs_hist)
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        # obs_hist_full = obs_hist_full[:,5:,:].view(b,-1)
        with torch.no_grad():
            latent = self.mlp_encoder(obs_hist_full[:,5:,:].reshape(b,-1))
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)
            # vel = self.history_encoder(obs_hist_full).detach()
            # #z = F.normalize(latents[:,3:],dim=-1,p=2).detach()
            # z = latents[:,3:].detach()
            # vel = latents[:,:3].detach()
        actor_input = torch.cat([vel.detach(),z.detach(),obs.detach()],dim=-1)
        mean  = self.actor(actor_input)
        # mean = self.actor(torch.cat([vel.detach(),z.detach()],dim=-1),obs.detach())
        return mean
    
    # def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
    #     obs = obs.detach()
    #     obs_hist = obs_hist.detach()
        
    #     b = obs.size()[0]

    #     obs_hist = obs_hist[:,5:,:].reshape(b,-1)

    #     latent = self.mlp_encoder(obs_hist)
    #     z1 = self.latent_layer(latent)
    #     vel = self.vel_layer(latent.detach())

    #     z2 = self.obs_encoder(obs)

    #     z1 = self.projector(z1) 
    #     z2 = self.projector(z2)

    #     c = self.bn(z1).T @ self.bn(z1)
    #     c.div_(b)

    #     on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    #     off_diag = off_diagonal(c).pow_(2).sum()

    #     priv_loss = F.mse_loss(vel,priv)

    #     loss = on_diag + weight*off_diag + priv_loss
        
    #     return loss

    def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
        obs,obs_hist = self.normalize(obs,obs_hist)

        obs = obs.detach()
        obs_hist = obs_hist.detach()

        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b = obs.size()[0]

        # obs_hist = obs_hist[:,5:,:].reshape(b,-1)

        z1 = self.mlp_encoder(obs_hist_full[:,5:,:].reshape(b,-1))
        z2 = self.mlp_encoder(obs_hist[:,5:,:].reshape(b,-1))

        z1_l = self.latent_layer(z1)
        z1_v = self.vel_layer(z1)

        z2_l = self.latent_layer(z2)
        # z2_v = z2[:,:3]

        # z1_l = F.normalize(z1_l,dim=-1,p=2)
        # z2_l = F.normalize(z2_l,dim=-1,p=2)

        z1_l = self.projector(z1_l) 
        z2_l = self.projector(z2_l)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        priv_loss = F.mse_loss(z1_v,priv)

        loss = on_diag + weight*off_diag + priv_loss
        
        return loss

def off_diagonal(x):
    n,m = x.shape
    assert n==m
    return x.flatten()[:-1].view(n-1,n+1)[:,1:].flatten()

class AeActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 encoder_dims,
                 decoder_dims,
                 actor_dims,
                 num_actions,
                 activation,
                 latent_dim) -> None:
        super(AeActor,self).__init__()
        self.ae = AutoEncoder(activation_fn=activation,
                            input_size=num_prop*num_hist,
                            encoder_dims=encoder_dims,
                            decoder_dims=decoder_dims,
                            latent_dim=latent_dim,
                            output_size=num_prop)
        
        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))

    def forward(self,obs,obs_hist):
        # self.rnn_encoder.reset_hidden()
        obs_hist_full = torch.cat([
                obs_hist[:,1:],
                obs.unsqueeze(1)
            ], dim=1)
        b,t,n = obs_hist_full.size()
        obs_hist_full = obs_hist_full.view(b,-1)
        latent = self.ae.encode(obs_hist_full)
        actor_input = torch.cat([latent,obs],dim=-1)
        mean  = self.actor(actor_input)
        return mean

    def predict_next_state(self,obs_hist):
        b,t,n = obs_hist.size()
        obs_hist_flatten = obs_hist.view(b,-1)
        latent = self.ae.encode(obs_hist_flatten)
        predicted = self.ae.decode(latent)
        return predicted,latent
        
class ActorCriticRMA(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        super(ActorCriticRMA, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        cost_dims = kwargs['num_costs']
        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with student actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 32)
        # actor_teacher_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim+self.scan_encoder_output_dim,num_actions,actor_hidden_dims,last_act=False)
        actor_teacher_layers = mlp_factory(activation,num_prop+priv_encoder_output_dim+32,num_actions,actor_hidden_dims,last_act=False)

        self.actor_teacher_backbone = nn.Sequential(*actor_teacher_layers)
        self.actor_student_backbone = CnnActor(num_prop=num_prop,
                                               num_hist=num_hist,
                                               num_actions=num_actions,
                                               priv_encoder_output_dim=priv_encoder_output_dim,
                                               actor_hidden_dims=actor_hidden_dims,
                                               activation=activation)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+32,cost_dims,critic_hidden_dims,last_act=False)
        cost_layers.append(nn.Softplus())
        self.cost = nn.Sequential(*cost_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def get_std(self):
        return self.std
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        if self.teacher_act:
            mean = self.act_teacher(obs)
        else:
            mean = self.act_student(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        if hasattr(self, "act_teacher"):
            return self.act_teacher(obs)
        elif hasattr(self, "act_student"):
            return self.act_student(obs)
        else:
            self.update_distribution(obs)
            return self.action_mean
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_student(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        hist = obs[:, -self.num_hist*self.num_prop:].view(-1,self.num_hist,self.num_prop)
        mean = self.actor_student_backbone(obs_prop,hist)
        return mean
    
    def act_teacher(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]

        # scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        hist_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,hist_latent], dim=1)
        mean = self.actor_teacher_backbone(backbone_input)
        return mean
        
    def evaluate(self, obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        hist_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,hist_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        hist_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent,hist_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
     
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs):
        with torch.no_grad():
            target_mean = self.act_teacher(obs)
        mean = self.act_student(obs)

        loss = F.mse_loss(mean,target_mean.detach())
        return loss
    
    def imitation_mode(self):
        self.actor_teacher_backbone.eval()
        self.scan_encoder.eval()
        self.priv_encoder.eval()
    
    def save_torch_jit_policy(self,path,device):
        obs_demo_input = torch.randn(1,self.num_prop).to(device)
        hist_demo_input = torch.randn(1,self.num_hist,self.num_prop).to(device)
        model_jit = torch.jit.trace(self.actor_student_backbone,(obs_demo_input,hist_demo_input))
        model_jit.save(path)

class ActorCriticBarlowTwins(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        super(ActorCriticBarlowTwins, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        cost_dims = kwargs['num_costs']
        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        # n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        self.num_obs = num_prop + num_scan + num_hist * num_prop + num_priv_latent
        self.obs_normalize = EmpiricalNormalization(self.num_obs)

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with teacher actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            # scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            # self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            # self.scan_encoder_output_dim = scan_encoder_dims[-1]
            scan_encoder_layers = mlp_factory(activation,num_scan,scan_encoder_dims[-1],scan_encoder_dims[:-1],last_act=False)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            print("not using scan encoder")
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 16)

        # #MlpBarlowTwinsActor
        self.actor_teacher_backbone = MlpBarlowTwinsActor(num_prop=num_prop-3,
                                      num_hist=5,
                                      num_actions=num_actions,
                                      actor_dims=[512,256,128],
                                      mlp_encoder_dims=[128,64],
                                      activation=activation,
                                      latent_dim=16,
                                      obs_encoder_dims=[128,64])
        
        print(self.actor_teacher_backbone)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)
        print(self.critic)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim,cost_dims,critic_hidden_dims,last_act=False)
        cost_layers.append(nn.Softplus())
        self.cost = nn.Sequential(*cost_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def get_std(self):
        return self.std
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.act_teacher(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        if hasattr(self, "act_teacher"):
            return self.act_teacher(obs)
        elif hasattr(self, "act_student"):
            return self.act_student(obs)
        else:
            self.update_distribution(obs)
            return self.action_mean
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_teacher(self,obs, **kwargs):
        # obs_prop = obs[:, :self.num_prop]
        # obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        obs_prop = obs[:, 3:self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)[:,:,3:]
        mean = self.actor_teacher_backbone(obs_prop,obs_hist)
        return mean
        
    def evaluate(self, obs, **kwargs):
        obs = self.obs_normalize(obs)

        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        #history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs = self.obs_normalize(obs)

        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        #history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs,imi_weight=1):
        # obs_prop = obs[:, :self.num_prop]
        # obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        obs_prop = obs[:, 3:self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        # contact = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + 4]
        # vel = obs_hist[:,-1,:3]

        # priv = torch.cat([contact,vel],dim=-1)
        priv = obs_hist[:,-1,:3]

        loss = self.actor_teacher_backbone.BarlowTwinsLoss(obs_prop,obs_hist[:,:,3:],priv,5e-3)
        # loss = self.actor_teacher_backbone.SimSiamLoss(obs_prop,obs_hist[:,:,3:],priv,scan)
        # loss = self.actor_teacher_backbone.VaeLoss(obs_prop,obs_hist[:,:,3:],priv,scan)
        # loss = self.actor_teacher_backbone.VaeLoss(obs_prop,obs_hist[:,:,3:],priv)
        #loss = self.actor_teacher_backbone.maeLoss(obs_prop,obs_hist,priv)
        # loss = recon_loss + kl_loss + mseloss
        return loss
    
    def imitation_mode(self):
        pass
    
    def save_torch_jit_policy(self,path,device):
        self.actor_teacher_backbone.eval()

        obs_demo_input = torch.randn(1,self.num_prop-3).half().to(device)
        hist_demo_input = torch.randn(1,self.num_hist,self.num_prop-3).half().to(device)
        model_jit = torch.jit.trace(self.actor_teacher_backbone,(obs_demo_input,hist_demo_input))
        model_jit.save(path)
        obs_demo_input = torch.randn(1,self.num_prop-3).to(device)
        hist_demo_input = torch.randn(1,self.num_hist,self.num_prop-3).to(device)
        torch_out = torch.onnx.export(self.actor_teacher_backbone,
                            (obs_demo_input,hist_demo_input),
                            "policy.onnx",
                            input_names=["nn_input0", "nn_input1"],
                            output_names=["nn_output"],
                            verbose=False,
                            opset_version=13,
                            export_params=True
                            )
        # print(torch_out)
