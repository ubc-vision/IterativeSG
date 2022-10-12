import os
import sys
import torch
import numpy as np
import logging 
import detectron2.utils.comm as comm
import time 
import datetime
import pickle
import itertools
import pycocotools.mask as mask_util
from collections import OrderedDict
from detectron2.utils.logger import setup_logger, log_every_n_seconds
from detectron2.engine import DefaultTrainer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    build_batch_data_loader
)
from detectron2.evaluation import DatasetEvaluators, DatasetEvaluator, inference_on_dataset, print_csv_format, inference_context
from imantics import Polygons, Mask

from detectron2.engine import hooks, HookBase
from ..data import DetrDatasetMapper
from detectron2.evaluation import (
    COCOEvaluator,
    SemSegEvaluator
)
from ..checkpoint import PeriodicCheckpointerWithEval
from ..evaluation import scenegraph_inference_on_dataset, SceneGraphEvaluator
from detectron2.engine import hooks
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.data.common import MapDataset, DatasetFromList
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import trivial_batch_collator
from detectron2.utils.comm import get_world_size, is_main_process
from typing import Any, Dict, List, Set
from detectron2.solver.build import maybe_add_gradient_clipping

class JointTransformerTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DetrDatasetMapper(cfg, True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DetrDatasetMapper(cfg, False))

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        # if comm.is_main_process():
        #     ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=1))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        ret.append(PeriodicCheckpointerWithEval(cfg.TEST.EVAL_PERIOD, test_and_save_results, self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=1))
        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    # def run_step(self):
    #     """
    #     Implement the AMP training logic.
    #     """
    #     assert self._trainer.model.training, "[AMPTrainer] model was changed to eval mode!"
    #     assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
    #     from torch.cuda.amp import autocast

    #     start = time.perf_counter()
    #     data = next(self._trainer._data_loader_iter)
    #     data_time = time.perf_counter() - start

    #     with autocast():
    #         loss_dict = self._trainer.model(data)
    #         if isinstance(loss_dict, torch.Tensor):
    #             losses = loss_dict
    #             loss_dict = {"total_loss": loss_dict}
    #         else:
    #             losses = sum(loss_dict.values())

    #     self._trainer.optimizer.zero_grad()
    #     self._trainer.grad_scaler.scale(losses).backward()
    #     for name, param in self._trainer.model.named_parameters():
    #         try:
    #             print (name, param.grad.norm())
    #         except:
    #             print (name, param.grad)
    #     import ipdb; ipdb.set_trace()
    #     self._write_metrics(loss_dict, data_time)

    #     self.grad_scaler.step(self.optimizer)
    #     self.grad_scaler.update()

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        logger = logging.getLogger("detectron2")
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            if "relation" in key:
                lr = lr * cfg.SOLVER.RELATION_MULTIPLIER
                logger.info("Setting LR for {} to {}".format(key, lr)) 
            if "detr.transformer.encoder" in key or "detr.transformer.decoder.layers" in key or "detr.query_embed" in key or 'backbone' in key or 'detr.transformer.decoder.norm' in key:
                lr = lr * cfg.SOLVER.ENTITY_MULTIPLIER
                logger.info("Setting LR for {} to {}".format(key, lr))
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)

        
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # import ipdb; ipdb.set_trace()
            
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            if cfg.MODEL.DETR.RELATION_HEAD and cfg.MODEL.META_ARCHITECTURE != 'DetrWithSGGBBox' and cfg.MODEL.META_ARCHITECTURE != 'Detr' and cfg.MODEL.META_ARCHITECTURE != 'QuerySplitObjectDetr' and cfg.MODEL.META_ARCHITECTURE != 'QuerySplitUnionBoxDetr' and cfg.MODEL.META_ARCHITECTURE != 'QuerySplitObjectDetrTest' and cfg.MODEL.META_ARCHITECTURE != 'RelationDetr' and cfg.MODEL.META_ARCHITECTURE != 'ConditionalDETR' and cfg.MODEL.META_ARCHITECTURE != 'QueryConditionalDETR' and cfg.MODEL.META_ARCHITECTURE != 'QueryConditionalDeformableDETR' and cfg.MODEL.META_ARCHITECTURE != 'LatentBoxConditionalDETR' and cfg.MODEL.META_ARCHITECTURE != 'LatentRelationDETRTest' and cfg.MODEL.META_ARCHITECTURE != 'LatentBoxDETR' and cfg.MODEL.META_ARCHITECTURE != 'LatentBoxDeformableDETR' and cfg.MODEL.META_ARCHITECTURE != 'LatentBoxDETRTest' and cfg.MODEL.META_ARCHITECTURE != 'LatentBoxConditionalDETRTest' and cfg.MODEL.META_ARCHITECTURE != 'LatentRelationCoordsDETRTest' and cfg.MODEL.META_ARCHITECTURE != 'LatentBoxCoordsDetr':
            #  and cfg.MODEL.META_ARCHITECTURE != 'LatentRelationCoordsNoAttentionDETR':
                evaluator = SceneGraphEvaluator(dataset_name, cfg, True, output_folder)
            else:
                evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
            results_i = scenegraph_inference_on_dataset(cfg, model, data_loader, evaluator)
            
            # print("Out of sg inference")
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
        comm.synchronize()
        if len(results) == 1:
            results = list(results.values())[0]
        return results
