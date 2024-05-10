import copy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from PIL import ImageColor, Image, ImageDraw, ImageFont
from masking_generator import CubeMaskGenerator
from kornia.augmentation import CenterCrop, RandomAffine, RandomCrop, RandomResizedCrop
from kornia.filters import GaussianBlur2d

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()

class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise

class MBR(nn.Module):
    def __init__(
        self,
        encoder,
        img_size,
        augmentation,
        aug_prob,
        mask_ratio,
        jumps,
        patch_size,
        block_size,
        device,
    ):
        super().__init__()
        self.aug_prob = aug_prob
        self.device = device
        self.jumps = jumps

        self.img_size = img_size
        input_size = img_size // patch_size
        # print('jumps', self.jumps)
        self.masker = CubeMaskGenerator(
            input_size=input_size,
            image_size=img_size,
            clip_size=self.jumps + 1,
            block_size=block_size,
            mask_ratio=mask_ratio,
        )  # 1 for mask, num_grid=input_size

        # self.position = PositionalEmbedding(encoder_feature_dim)
        # self.position = nn.Parameter(torch.zeros(1, jumps+1, encoder_feature_dim))

        # self.state_mask_token = nn.Parameter(torch.zeros(1, 1, encoder_feature_dim))
        # self.action_mask_token = nn.Parameter(torch.zeros(1, 1, encoder_feature_dim))

        # self.state_flag = nn.Parameter(torch.zeros(1, 1, encoder_feature_dim))
        # self.action_flag = nn.Parameter(torch.zeros(1, 1, encoder_feature_dim))

        self.encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)

        # self.transformer = nn.ModuleList(
        #     [
        #         Block(
        #             encoder_feature_dim,
        #             num_heads,
        #             mlp_ratio=2.0,
        #             qkv_bias=False,
        #             qk_scale=None,
        #             drop=0.0,
        #             attn_drop=0.0,
        #             drop_path=0.0,
        #             init_values=0.0,
        #             act_layer=nn.GELU,
        #             norm_layer=nn.LayerNorm,
        #             attn_head_dim=None,
        #         )
        #         for _ in range(num_attn_layers)
        #     ]
        # )
        # self.action_emb = nn.Linear(action_shape[0], encoder_feature_dim)
        # self.action_predictor = nn.Sequential(
        #     nn.Linear(encoder_feature_dim, encoder_feature_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(encoder_feature_dim * 2, action_shape[0]),
        # )

        """ Data augmentation """
        self.intensity = Intensity(scale=0.05)
        self.transforms = []
        self.eval_transforms = []
        self.uses_augmentation = True
        for aug in augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (0.14, 0.14), (0.9, 1.1), (-5, 5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "crop":
                transformation = RandomCrop((84, 84))
                # Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
                eval_transformation = CenterCrop((84, 84))
                self.uses_augmentation = True
                imagesize = 84
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "shift":
                transformation = nn.Sequential(
                    nn.ReplicationPad2d(4), RandomCrop((84, 84))
                )
                eval_transformation = nn.Identity()
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

        self.apply(self._init_weights)
        # trunc_normal_(self.position, std=.02)
        # trunc_normal_(self.state_mask_token, std=0.02)
        # trunc_normal_(self.action_mask_token, std=0.02)
        # trunc_normal_(self.state_flag, std=.02)
        # trunc_normal_(self.action_flag, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = tools.maybe_transform(
                    image, transform, eval_transform, p=self.aug_prob
                )
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):
        images = images.float() / 255.0 if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(
                self.transforms, self.eval_transforms, flat_images
            )
        else:
            processed_images = self.apply_transforms(
                self.eval_transforms, None, flat_images
            )
        processed_images = processed_images.view(
            *images.shape[:-3], *processed_images.shape[1:]
        )
        return processed_images

    def spr_loss(self, latents, target_latents, no_grad=False):
        if no_grad:
            with torch.no_grad():
                global_latents = latents
        else:
            global_latents = latents

        with torch.no_grad():
            global_targets = target_latents
        # targets = global_targets.view(-1, observation.shape[1], self.jumps + 1,
        #                               global_targets.shape[-1]).transpose(
        #                                   1, 2)
        # latents = global_latents.view(-1, observation.shape[1], self.jumps + 1,
        #                               global_latents.shape[-1]).transpose(
        #                                   1, 2)
        # loss = self.norm_mse_loss(latents, targets, mean=False)
        loss = self.norm_mse_loss(global_latents, global_targets, mean=False).mean()
        # split to [jumps, bs]
        # return loss.view(-1, observation.shape[1])
        return loss

    def norm_mse_loss(self, f_x1s, f_x2s, mean=True):
        f_x1 = F.normalize(
            f_x1s.float(), p=2.0, dim=-1, eps=1e-3
        )  # (bs*(1+jumps), 512)
        f_x2 = F.normalize(f_x2s.float(), p=2.0, dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1)
        loss = loss.mean(0) if mean else loss
        return loss

class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.MBR = MBR(
            encoder=self.encoder,
            img_size=config.size[0],
            augmentation=config.augmentation,
            aug_prob=config.aug_prob,
            mask_ratio=config.mask_ratio,
            jumps=config.batch_length-1, # TODO ???
            patch_size=config.patch_size,
            block_size=config.block_size,
            device=config.device
        ).to(config.device)
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_input_layers,
            config.dyn_output_layers,
            config.dyn_rec_depth,
            config.dyn_shared,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_temp_post,
            config.dyn_min_std,
            config.dyn_cell,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
            config.dyn_cont_stoch_size,
        )
        self.target_dynamics = copy.deepcopy(self.dynamics)
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        if config.reward_head == "symlog_disc":
            self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                (255,),
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.heads["reward"] = networks.MLP(
                feat_size,  # pytorch version
                [],
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
                device=config.device,
            )
        self.heads["cont"] = networks.MLP(
            feat_size,  # pytorch version
            [],
            config.cont_layers,
            config.units,
            config.act,
            config.norm,
            dist="binary",
            device=config.device,
        )
        for name in config.grad_heads:
            assert name in self.heads, name
            
        # self.decoder = networks.MultiDecoder(
        #     self.embed_size, shapes, **config.decoder
        # )
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        self._scales = dict(reward=config.reward_scale, cont=config.cont_scale, simsr=config.simsr_scale, mbr=config.mbr_scale)
        
    def update_SimSR(self, mask_post_feat, true_post_feat, mask_prior_feat, true_prior_feat, data):
        l2_norm = lambda x: nn.functional.normalize(x, dim=-1, p=2)
        # (batch_size, batch_length, ...)
        
        # with torch.no_grad():
            # post_tar, _ = self.dynamics_target.observe(
            #     embed, data["action"], data["is_first"]
            # )
            # next_prior = self.dynamics.img_step(post, data["action"], sample=self._config.imag_sample) # succ : {'stoch' : torch.Size([1024, 32, 32]), 'deter' : torch.Size([1024, 512]), 'logit' : torch.Size([1024, 32, 32])}
        
        
        
        z_a = l2_norm(mask_post_feat[:, :-1])
        z_b = l2_norm(true_post_feat[:, :-1])
        pred_a = l2_norm(mask_prior_feat[:, 1:])
        pred_b = l2_norm(true_prior_feat[:, 1:])
        reward = data["reward"][:, :-1]
        if self._config.task == "rms_push_cube":
            reward = reward / 6.5 * 2
        elif self._config.task == "rms_lift_cube":
            reward = reward / 6.5 * 2
        elif self._config.task == "rms_turn_faucet":
            reward = reward / 10 * 2
        elif self._config.task == "rms_peg_insertion_side":
            reward = reward / 25 * 2
        elif self._config.task == "rms_assembling_kits":
            reward = reward / 10 * 2
        elif self._config.task == "rms_plug_charger":
            reward = reward / 50 * 2
        elif self._config.task == "rms_panda_avoid_obstacles":
            reward = reward / 10 * 2
        if not reward.min() >= 0 and reward.max() <= 2:
            print(f"Reward range out of [0, 2], min reward = {reward.min()}, max reward = {reward.max()}")
        
        z_a, pred_a, reward = z_a.reshape(-1, z_a.shape[-1]), pred_a.reshape(-1, pred_a.shape[-1]), reward.reshape(-1, 1)
        z_b, pred_b = z_b.reshape(-1, z_b.shape[-1]), pred_b.reshape(-1, pred_b.shape[-1])
        
        def compute_dis(features_a, features_b):
            similarity_matrix = torch.matmul(features_a, features_b.T)
            dis = 1-similarity_matrix
            return dis
        
        r_diff = torch.abs(reward.T - reward)
        next_diff = compute_dis(pred_a, pred_b)
        z_diff = compute_dis(z_a, z_b)
        bisimilarity = r_diff + self._config.simsr_discount * next_diff
        loss = torch.nn.HuberLoss()(z_diff, bisimilarity.detach())
        return loss

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)
        with torch.no_grad():
            true_embed = self.MBR.target_encoder(data)
        
        mask = self.MBR.masker()
        mask = mask[:, None].expand(mask.size(0), data['image'].shape[0], *mask.size()[1:]).transpose(-3, -1).transpose(0, 1)
        masked_image = data['image'] * (1 - mask.float().to(self._config.device))
        masked_image = self.MBR.transform(masked_image, augment=True)
        data['image'] = masked_image
        
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.MBR.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                feat = self.dynamics.get_feat(post)
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    inp = feat if grad_head else feat.detach()
                    pred = head(inp)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                # preds.update(self.decoder(embed))
                losses = {}
                for name, pred in preds.items():
                    if not self._config.dyn_discrete and name == 'reward':
                        like = pred.log_prob(data[name].unsqueeze(-1))
                    else:
                        like = pred.log_prob(data[name])
                    losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                prior_feat = self.dynamics.get_feat(prior)
                post_cont_feat = self.dynamics.get_mlp_feat(feat)
                prior_cont_feat = self.dynamics.get_mlp_feat(prior_feat)
                with torch.no_grad():
                    init_state = self.target_dynamics.initial(data["action"].shape[0])
                    prev_post = {key : torch.cat([init_state[key].unsqueeze(1), value], dim=1)[:, :-1] for key, value in post.items()}
                    true_post, true_prior = self.target_dynamics.obs_step_by_prior(prev_post, prior, data["action"], true_embed, data["is_first"])
                    true_post_feat = self.target_dynamics.get_feat(true_post)
                    true_prior_feat = self.target_dynamics.get_feat(true_prior)
                    true_post_cont_feat = self.target_dynamics.get_mlp_feat(true_post_feat)
                if not self._config.nomlr:
                    losses["mbr"] = self.MBR.spr_loss(post_cont_feat, true_post_cont_feat) * self._scales.get("mbr", 1.0)
                if not self._config.nosimsr:
                    losses['simsr'] = self.update_SimSR(feat, true_post_feat, prior_feat, true_prior_feat, data) * self._scales.get("simsr", 1.0)
                model_loss = sum(losses.values()) + kl_loss
            metrics = self._model_opt(model_loss, self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        metrics["min_reward"] = to_np(data['reward'].min())
        metrics["max_reward"] = to_np(data['reward'].max())
        with torch.cuda.amp.autocast(self._use_amp):
            # assert "update MRB encoder"
            tools.soft_update_params(self.MBR.encoder, self.MBR.target_encoder, self._config.encoder_tau)
            tools.soft_update_params(self.dynamics, self.target_dynamics, self._config.dynamics_tau)
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        true_post = {k: v.detach() for k, v in true_post.items()}
        return true_post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        if "image" in obs:
            obs["image"] = torch.Tensor(obs["image"]) / 255.0 - 0.5
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    # def video_pred(self, data):
    #     data = self.preprocess(data)
    #     embed = self.encoder(data)

    #     model = self.decoder(embed[:6])["image"].mode()[
    #         :6
    #     ]

    #     truth = data["image"][:6] + 0.5
    #     model = model + 0.5
    #     error = (model - truth + 1.0) / 2.0

    #     return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.ActionHead(
            feat_size,
            config.num_actions,
            config.actor_layers,
            config.units,
            config.act,
            config.norm,
            config.actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=1.0,
            unimix_ratio=config.action_unimix_ratio,
        )
        if config.value_head == "symlog_disc":
            self.value = networks.MLP(
                feat_size,
                (255,),
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
                device=config.device,
            )
        else:
            self.value = networks.MLP(
                feat_size,
                [],
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
                device=config.device,
            )
        if config.slow_value_target:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective=None,
        action=None,
        reward=None,
        imagine=None,
        tape=None,
        repeats=None,
    ):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, repeats
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_state,
                    imag_action,
                    target,
                    actor_ent,
                    state_ent,
                    weights,
                    base,
                )
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.slow_value_target:
                    value_loss = value_loss - value.log_prob(
                        slow_target.mode().detach()
                    )
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor_dist in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, states, actions

    def _compute_target(
        self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
    ):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        if self._config.future_entropy and self._config.actor_entropy > 0:
            reward += self._config.actor_entropy * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy > 0:
            reward += self._config.actor_state_entropy * state_ent
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_state,
        imag_action,
        target,
        actor_ent,
        state_ent,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(inp)
        actor_ent = policy.entropy()
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            values = self.reward_ema.values
            metrics["EMA_005"] = to_np(values[0])
            metrics["EMA_095"] = to_np(values[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        if not self._config.future_entropy and self._config.actor_entropy > 0:
            actor_entropy = self._config.actor_entropy * actor_ent[:-1][:, :, None]
            actor_target += actor_entropy
        if not self._config.future_entropy and (self._config.actor_state_entropy > 0):
            state_entropy = self._config.actor_state_entropy * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        actor_loss = -torch.mean(weights[:-1] * actor_target)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
