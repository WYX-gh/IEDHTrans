from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import numpy as np
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn,MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.network_architecture.IEDHTrans import TokenSeg
from torch.cuda.amp import GradScaler
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.network_architecture.ContrastiveLoss import HybridContrastiveLoss


class IEDHTransTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = True  # 启用深度监督
        self.contrastive_weight = 0.1 #根据实验结果进行调整
        self.dice_weight = 0.45
        self.ce_weight = 0.45

        if self.device.type == 'cuda':
            self.grad_scaler = GradScaler()
        else:
            self.grad_scaler = None 


        self.contrastive_criterion = HybridContrastiveLoss(
            cross_modality_weight=0.5,
            cross_scale_weight=0.5,
            temperature=0.5
        ).to(self.device)

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({}, {'batch_dice': self.configuration_manager.batch_dice, 
                                  'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                           use_ignore_label=self.label_manager.ignore_label is not None,
                           dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                           'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, 
                          weight_ce=1, weight_dice=1,
                          ignore_label=self.label_manager.ignore_label, 
                          dice_class=MemoryEfficientSoftDiceLoss)

        # 深监督损失
        deep_supervision_scales =self._get_deep_supervision_scales()
        if deep_supervision_scales is not None:
            weights = [1 / (2 ** i) for i in range(len(deep_supervision_scales))]
            weights = weights / np.sum(weights)
            loss = DeepSupervisionWrapper(loss, weights)
        else:
            loss = loss

        return loss

    def initialize(self):
        """初始化自定义模型结构"""
        if not self.was_initialized:
            # 确定输入通道数
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )
            
            # 初始化PLHN模型
            self.network = TokenSeg(
                inch=self.num_input_channels,  # 动态适配输入通道
                outch=self.label_manager.num_segmentation_heads,
                base_channel=32,
                hidden_size=256,
                imgsize=self.configuration_manager.patch_size,  # 使用配置的patch size
                TransformerLayerNum=3
            ).to(self.device)


            # 编译模型（如果启用）
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            # 初始化优化器
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            
            # DDP处理
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            # 初始化损失函数
            self.loss = self._build_loss()
            self.was_initialized = True

            self.print_to_log_file("="*50)
            self.print_to_log_file("Thanks nnUNet" )
            self.print_to_log_file("now using IEDHTrans model")
            self.print_to_log_file("="*50)
        
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device) for t in target]
        else:
            target = target.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        # 前向传播
        with torch.autocast(self.device.type, enabled=(self.device.type == 'cuda')):
            output = self.network(data)


            seg_outputs = output['seg_output']
            contrastive_features = output['features']

            #print("seg_outputs shape:", [output.shape for output in seg_outputs])
            #print("target length:", len(target))
            #for i in range(len(seg_outputs)):
            #    print(f"target[{i}] shape:", target[i].shape)

            seg_loss = self.loss(seg_outputs, target)

            


            contrastive_loss = self.contrastive_criterion(contrastive_features)
            total_loss = (
                self.dice_weight * seg_loss + 
                self.contrastive_weight * contrastive_loss
            )


            #l = self.loss(output, target)

        # 反向传播
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            #self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        #return {'loss': l.detach().cpu().numpy()}
        return {
            'loss': total_loss.detach().cpu().numpy(),
            'total_loss': total_loss.detach().cpu().numpy(),
            'seg_loss': seg_loss.detach().cpu().numpy(),
            'contrastive_loss': contrastive_loss.detach().cpu().numpy()
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device) for t in target]
        else:
            target = target.to(self.device)

        # 前向传播
        with torch.no_grad():
            output = self.network(data)
            seg_outputs = output['seg_output']
            #l = self.loss(output, target)
            l = self.loss(seg_outputs, target)

        # 计算验证指标
        #axes = [0] + list(range(2, output[0].ndim))
        axes = [0] + list(range(2, seg_outputs[0].ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(seg_outputs[0]) > 0.5).long()
        else:
            output_seg = seg_outputs[0].argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(seg_outputs[0].shape, device=seg_outputs[0].device)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target[0], axes=axes)
        tp_hard = tp.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        }