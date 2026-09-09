# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics
from logger import format_flat_train_log, format_eval_metrics, get_lr

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.run_name = None
        self.enable_tensorboard = False
        self.save_aux_artifacts = False
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()  # move model to GPU

        # get current time
        self.timenow = time_now
        # create directory path
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        if not self.enable_tensorboard:
            return NullWriter()
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]


    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']],find_unused_parameters=True, output_device=self.config['local_rank'])
            #self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key,ckpt_info=None):
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        save_path = os.path.join(self.log_dir, "model_best.pth")
        payload = {
            'state_dict': model_to_save.state_dict(),
            'ckpt_info': ckpt_info,
            'phase': phase,
            'dataset_key': dataset_key,
        }
        torch.save(payload, save_path)
        self.logger.info(
            f"Best model checkpoint saved: path={save_path}, current ckpt is {ckpt_info}"
        )

    def save_latest_ckpt(self, epoch, iteration):
        if not self.config.get('save_ckpt', True) or self.config.get('local_rank', 0) != 0:
            return
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        payload = {
            'state_dict': model_to_save.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'epoch': epoch,
            'iteration': iteration,
            'best_metrics_all_time': {
                dataset_key: dict(metric_values)
                for dataset_key, metric_values in self.best_metrics_all_time.items()
            },
        }
        save_path = os.path.join(self.log_dir, "latest.pth")
        torch.save(payload, save_path)
        self.logger.info(f"Latest checkpoint saved before evaluation: path={save_path}")

    def save_smoke_ckpt(self, epoch, iteration):
        if not self.config.get('save_ckpt', True) or self.config.get('local_rank', 0) != 0:
            return
        self.save_latest_ckpt(epoch, iteration)
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        save_path = os.path.join(self.log_dir, "model_best.pth")
        torch.save(
            {
                'state_dict': model_to_save.state_dict(),
                'ckpt_info': f"smoke_{epoch}+{iteration}",
                'phase': 'smoke',
                'dataset_key': 'smoke',
            },
            save_path,
        )
        self.logger.info(f"Smoke checkpoint saved: path={save_path}")

    def save_swa_ckpt(self):
        if self.swa_model is None:
            return
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa_model.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        model_to_save = self.swa_model.module if hasattr(self.swa_model, 'module') else self.swa_model
        payload = {
            'state_dict': model_to_save.state_dict(),
            'phase': 'swa',
            'dataset_key': 'swa',
        }
        torch.save(payload, save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")


    def save_feat(self, phase, fea, dataset_key):
        if not self.save_aux_artifacts:
            return
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        if not self.save_aux_artifacts:
            return
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        if not self.save_aux_artifacts:
            return
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self,data_dict):
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:

            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            self.optimizer.zero_grad()
            losses['overall'].backward()
            #self.model.module.set_mask_grad()
            self.optimizer.step()


            return losses,predictions


    def train_epoch(
        self,
        epoch,
        train_data_loader,
        test_data_loaders=None,
        ):

        if self.config['ddp'] and hasattr(train_data_loader, 'sampler') and hasattr(train_data_loader.sampler, 'set_epoch'):
            train_data_loader.sampler.set_epoch(epoch)
        test_step = max(1, len(train_data_loader))    # evaluate once at the end of each epoch
        step_cnt = epoch * len(train_data_loader)

        # save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)
        test_best_metric = None
        test_current_metric = None
        log_interval = max(1, int(self.config.get('rec_iter', 100)))
        max_train_iters = self.config.get('max_train_iters')
        max_train_iters = int(max_train_iters) if max_train_iters else None
        last_log_time = time.time()

        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=len(train_data_loader)):
            self.setTrain()
            # more elegant and more scalable way of moving data to GPU
            for key in data_dict.keys():
                if data_dict[key]!=None and key!='name':
                    data_dict[key]=data_dict[key].cuda()

            losses,predictions=self.train_step(data_dict)

            # update learning rate
            if self.scheduler is not None and self.config.get('scheduler_step_unit', 'epoch') == 'iteration':
                swa_enabled = self.config.get('swa_enabled', False)
                swa_start = self.config.get('swa_start_epoch', max(1, self.config['nEpochs'] // 2))
                if not (swa_enabled and epoch >= swa_start):
                    self.scheduler.step()

            if 'SWA' in self.config and self.config['SWA'] and epoch>self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if ((iteration + 1) % log_interval == 0 or iteration == 0) and self.config['local_rank']==0:
                if self.config['SWA'] and (epoch>self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()
                loss_log_values = {}
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        continue
                    loss_log_values[k] = v_avg
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                metric_log_values = {}
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        continue
                    metric_log_values[k] = v_avg
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                now = time.time()
                iter_time = (now - last_log_time) / max(1, log_interval if iteration != 0 else 1)
                last_log_time = now
                self.logger.info(
                    format_flat_train_log(
                        iteration=step_cnt,
                        iter_time=iter_time,
                        lr=get_lr(self.optimizer),
                        loss_values=loss_log_values,
                        metric_values=metric_log_values,
                    )
                )



                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            # run test
            #if True:
            if (step_cnt+1) % test_step == 0:
                if test_data_loaders is not None and (not self.config['ddp'] ):
                    self.save_latest_ckpt(epoch, iteration)
                    self.logger.info("***** Evaluation *****")
                    test_current_metric, test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                elif test_data_loaders is not None and self.config['ddp']:
                    if dist.get_rank() == 0:
                        self.save_latest_ckpt(epoch, iteration)
                    dist.barrier()
                    if dist.get_rank() == 0:
                        self.logger.info("***** Evaluation *****")
                    test_current_metric, test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    dist.barrier()
                else:
                    test_current_metric = None
                    test_best_metric = None

                    # total_end_time = time.time()
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
            step_cnt += 1
            if max_train_iters is not None and (iteration + 1) >= max_train_iters:
                self.save_smoke_ckpt(epoch, iteration)
                if self.config['ddp']:
                    dist.barrier()
                if self.config.get('local_rank', 0) == 0:
                    self.logger.info("Reached max_train_iters=%d; ending training early.", max_train_iters)
                break
        return {'current': test_current_metric, 'best': test_best_metric}

    def get_respect_acc(self,prob,label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        return acc_real,acc_fake

    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists = []
        collect_features = bool(self.config.get('save_feat', False))
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader),total=len(data_loader)):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key]!=None:
                    data_dict[key]=data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            if collect_features and 'feat' in predictions:
                feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                with torch.no_grad():
                    if type(self.model) is DDP:
                        losses = self.model.module.get_losses(data_dict, predictions)
                    else:
                        losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        feature_array = np.array(feature_lists) if collect_features else np.empty((0,), dtype=np.float32)
        image_indices, image_names = self._get_loader_image_indices_and_names(data_loader, len(label_lists))
        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists), feature_array, image_indices, image_names

    def _get_loader_image_indices_and_names(self, data_loader, num_items):
        dataset = data_loader.dataset
        sampler = getattr(data_loader, 'sampler', None)
        if sampler is None:
            indices = range(len(dataset))
        else:
            indices = list(iter(sampler))
        indices = list(indices)[:num_items]
        return indices, [dataset.data_dict['image'][idx] for idx in indices]

    def _recorder_to_payload(self, recorder_dict):
        return {
            key: {'sum': recorder.sum, 'num': recorder.num}
            for key, recorder in recorder_dict.items()
        }

    def _payload_to_recorder(self, payloads):
        merged = defaultdict(Recorder)
        for payload in payloads:
            for key, value in payload.items():
                merged[key].sum += value['sum']
                merged[key].num += value['num']
        return merged

    def _gather_eval_outputs(self, losses, predictions, labels, features, image_indices, image_names):
        if not (self.config.get('ddp', False) and dist.is_available() and dist.is_initialized()):
            return losses, predictions, labels, features, image_names

        local_payload = {
            'losses': self._recorder_to_payload(losses),
            'predictions': predictions,
            'labels': labels,
            'features': features,
            'image_indices': image_indices,
            'image_names': image_names,
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_payload)

        if dist.get_rank() != 0:
            return None, None, None, None, None

        gathered_losses = self._payload_to_recorder([item['losses'] for item in gathered])
        gathered_predictions = np.concatenate([item['predictions'] for item in gathered])
        gathered_labels = np.concatenate([item['labels'] for item in gathered])
        if any(item['features'].size > 0 for item in gathered):
            gathered_features = np.concatenate([item['features'] for item in gathered if item['features'].size > 0])
        else:
            gathered_features = features
        gathered_indices = []
        gathered_names = []
        for item in gathered:
            gathered_indices.extend(item['image_indices'])
            gathered_names.extend(item['image_names'])

        order = np.argsort(np.array(gathered_indices))
        gathered_predictions = gathered_predictions[order]
        gathered_labels = gathered_labels[order]
        if gathered_features.size > 0:
            gathered_features = gathered_features[order]
        gathered_names = [gathered_names[idx] for idx in order]
        return gathered_losses, gathered_predictions, gathered_labels, gathered_features, gathered_names

    def save_best(self,epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            self.best_metrics_all_time[key]['epoch'] = epoch
            self.best_metrics_all_time[key]['iteration'] = iteration
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            loss_pieces = [f"eval/dataset={key}", f"eval/step={step}"]
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg == None:
                    continue
                writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
                loss_pieces.append(f"loss/{k}={v_avg:.6f}")
            self.logger.info(", ".join(loss_pieces))
        metric_str = format_eval_metrics(key, metric_one_dataset)
        metric_pieces = [f"eval/step={step}", metric_str]
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k == 'dataset_dict':
                continue
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            writer.add_scalar(f'test_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'test_metrics/acc_fake', acc_fake, global_step=step)
            metric_pieces.append(f"acc_real={acc_real:.6f}")
            metric_pieces.append(f"acc_fake={acc_fake:.6f}")
        self.logger.info(", ".join(metric_pieces))

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        # set model to eval mode
        self.setEval()
        if self.run_name is not None:
            self.logger.info(f"Run name: {self.run_name}")

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        current_metrics = defaultdict(dict)
        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps, image_indices, image_names = self.test_one_dataset(test_data_loaders[key])
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps, image_names = self._gather_eval_outputs(
                losses_one_dataset_recorder,
                predictions_nps,
                label_nps,
                feature_nps,
                image_indices,
                image_names,
            )
            if self.config.get('ddp', False) and dist.get_rank() != 0:
                continue
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=image_names)
            current_metrics[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset)

        if self.config.get('ddp', False) and dist.get_rank() != 0:
            return None, None

        if len(keys)>0 and self.config.get('save_avg',False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            current_metrics['avg'] = deepcopy(avg_metric)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('Evaluation done.')
        return current_metrics, self.best_metrics_all_time  # current metrics and best-so-far metrics

    @torch.no_grad()
    def inference(self, data_dict):
        # Evaluation uses explicit result gathering, so call the wrapped module
        # directly instead of issuing DDP forward collectives.
        model = self.model.module if isinstance(self.model, DDP) else self.model
        predictions = model(data_dict, inference=True)
        return predictions
