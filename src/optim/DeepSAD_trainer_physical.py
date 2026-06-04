from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from torch.nn import MSELoss
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, hamming_loss

import setting
import wandb
import logging
import time
import torch
import torch.optim as optim
import numpy as np
import copy


def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def pairwise_apply(A: torch.Tensor, f):
    """
    Returns F with shape (M, M) where F[i, j] = f(A[i], A[j]).
    """
    f_i = torch.vmap(f, in_dims=(0, None))
    F   = torch.vmap(f_i, in_dims=(None, 0))(A, A)
    return F  # (M, M)

def cos_sim(u, v):
    den = (torch.norm(u) * torch.norm(v)).clamp_min(1e-8)
    return torch.dot(u, v) / den

def l_contrastive(A, same_mask):
    """
    Multi-label contrastive loss.

    :param A:         (k, k) matrix — pairwise cosine similarities scaled by tau.
    :param same_mask: (k, k) bool — True where two samples share ≥1 active anomaly class
                      (or are both normal).  Computed by the caller.
    :returns: scalar loss
    """
    k = A.size(0)
    device = A.device

    eye      = torch.eye(k, dtype=torch.bool, device=device)
    pos_mask = same_mask & ~eye    # positives: same class, not self
    neg_mask = ~same_mask          # negatives: different class

    # logsumexp over all non-self pairs (denominator)
    logits_not_self = A.masked_fill(eye, float('-inf'))
    lse_not_self    = torch.logsumexp(logits_not_self, dim=1)   # (k,)

    # average positive logit per anchor
    pos_counts = pos_mask.sum(dim=1)
    valid_pos  = pos_counts > 0
    sum_pos    = (A * pos_mask.float()).sum(dim=1)
    mean_pos   = torch.zeros_like(sum_pos)
    mean_pos[valid_pos] = sum_pos[valid_pos] / pos_counts[valid_pos]

    # anchors that also have negatives
    valid_neg = neg_mask.sum(dim=1) > 0

    valid = valid_pos & valid_neg
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss_i = lse_not_self - mean_pos
    return loss_i[valid].mean()


def get_all_centroids(A: torch.Tensor, y: torch.Tensor):
    labels = torch.unique(y)
    centroids = {}
    for label in labels:
        mask = (y == label)
        count = mask.sum()
        if count > 0:
            centroid = (A * mask.unsqueeze(1)).sum(dim=0) / count
            centroids[label.item()] = centroid
        else:
            centroids[label.item()] = torch.zeros_like(A[0])
    return centroids


class DeepSADTrainerPhysical(BaseTrainer):

    def __init__(self, n_known_outlier_classes: int, known_outlier_classes, outlier_classes,
                 coeff: dict, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6,
                 device: str = 'cuda', n_jobs_dataloader: int = 0, tau=0.1,
                 aug_mode: str = 'gaussian', nngmix_cfg: dict | None = None):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        self.n_known_outlier_classes = n_known_outlier_classes
        self.known_outlier_classes   = list(known_outlier_classes)
        self.outlier_classes         = list(outlier_classes)
        # Maps known anomaly class number → column index in multi-hot tensors
        self.known_col_indices = [self.outlier_classes.index(k) for k in self.known_outlier_classes]
        self.class_ids = [0] + list(self.known_outlier_classes)

        self.centroids = {
            'c_normal': torch.zeros((setting.rep,), device=self.device)
        }
        for i in range(n_known_outlier_classes):
            self.centroids[f'c_outlier_{i+1}'] = torch.zeros((setting.rep,), device=self.device)

        self.roc_curve = None
        self.coeff = coeff
        self.n_aug = 2
        self.tau   = tau

        self.eps      = 1e-6
        self.MSE_loss = MSELoss()

        # Augmentation mode: 'gaussian' | 'nngmix' | 'both'
        self.aug_mode = aug_mode
        _DEFAULT_NNG = {
            'nn_k': 10, 'nn_k_anomaly': 10,
            'mixup_alpha': 0.2, 'mixup_beta': 0.2,
            'nn_mix_gaussian': True, 'nn_mix_gaussian_std': 0.01,
            'use_uniform': False,
        }
        self._nng_cfg          = {**_DEFAULT_NNG, **(nngmix_cfg or {})}
        self._nng_X_anomaly    = None
        self._nng_sn_anomaly   = None
        self._nng_X_pool       = None
        self._nng_sn_pool      = None
        self._nng_tree_normal  = None
        self._nng_tree_anomaly = None

        self.per_class_thresholds = None   # list[float], one per known outlier class

        self.train_time          = None
        self.test_auc            = None
        self.test_time           = None
        self.test_scores         = None
        self.test_f1_macro_mh    = None
        self.test_f1_micro_mh    = None
        self.test_hamming_acc    = None
        self.test_subset_acc     = None

    # ------------------------------------------------------------------
    # Helper: boolean masks from multi-hot semi_targets
    # ------------------------------------------------------------------
    def _labeled_masks(self, semi_targets):
        """
        semi_targets: (batch, n_anomaly_classes) float tensor.
          all -1  → unlabeled
          all  0  → labeled normal
          some 1s → labeled anomaly
        Returns three (batch,) bool tensors: is_labeled_normal, is_labeled_anomaly, is_labeled.
        """
        is_unlabeled      = (semi_targets == -1).all(dim=1)
        is_labeled        = ~is_unlabeled
        is_labeled_normal = is_labeled & (semi_targets == 0).all(dim=1)
        is_labeled_anomaly = is_labeled & (semi_targets > 0).any(dim=1)
        return is_labeled_normal, is_labeled_anomaly, is_labeled

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------
    def loss_all(self, outputs, semi_targets, signal_pred, signal_next):
        loss = 0.0
        is_labeled_normal, is_labeled_anomaly, is_labeled = self._labeled_masks(semi_targets)

        # -------- Original Deep SAD loss --------
        dist = torch.sum((outputs - self.centroids['c_normal']) ** 2, dim=1)

        if is_labeled_normal.any():
            loss_normal = torch.mean(dist[is_labeled_normal])
        else:
            loss_normal = torch.tensor(0.0, device=self.device)

        if is_labeled_anomaly.any():
            loss_anomaly = torch.mean(torch.reciprocal(dist[is_labeled_anomaly] + self.eps))
        else:
            loss_anomaly = torch.tensor(0.0, device=self.device)

        loss_sad = loss_normal + loss_anomaly

        # -------- Physics-informed loss --------
        loss_pred = self.MSE_loss(signal_pred, signal_next)

        # -------- Directional / contrastive loss --------
        if is_labeled.sum() <= 1:
            loss_dir = torch.tensor(0.0, device=self.device)
        else:
            y = semi_targets[is_labeled].float()        # (k, n_anomaly_classes)

            # "same" = both normal OR sharing ≥1 active anomaly class
            # e.g. A={1,2}, B={1} → A·B=1>0 → same; A={1,2}, C={3} → A·C=0 → different
            is_normal_vec = (y == 0).all(dim=1)                                      # (k,)
            both_normal   = is_normal_vec.unsqueeze(0) & is_normal_vec.unsqueeze(1)  # (k, k)
            shared_class  = torch.mm(y, y.T) > 0                                     # (k, k)
            same_mask     = both_normal | shared_class

            A = pairwise_apply(outputs[is_labeled], cos_sim)
            loss_dir = l_contrastive(A / self.tau, same_mask)

        # -------- Clustering loss --------
        # Labeled normals → pull toward c_normal
        dist_normal = torch.sum((outputs[is_labeled_normal] - self.centroids['c_normal']) ** 2, dim=1)
        loss_normal_cl = dist_normal.mean() if is_labeled_normal.any() else torch.tensor(0.0, device=self.device)

        # Labeled known anomalies → each sample pulled toward its class's centroid
        loss_outlier = torch.tensor(0.0, device=self.device)
        for idx_k, col in enumerate(self.known_col_indices):
            mask_k = (semi_targets[:, col] == 1)
            if mask_k.any():
                dist_k = torch.sum(
                    (outputs[mask_k] - self.centroids[f'c_outlier_{idx_k+1}']) ** 2, dim=1)
                loss_outlier = loss_outlier + dist_k.mean()

        # Unlabeled samples → pull toward closest centroid
        is_unlabeled = ~is_labeled
        if is_unlabeled.any():
            n_c = self.n_known_outlier_classes + 1
            dist_unlabeled = torch.zeros(is_unlabeled.sum(), n_c, device=self.device)
            dist_unlabeled[:, 0] = torch.sum(
                (outputs[is_unlabeled] - self.centroids['c_normal']) ** 2, dim=1)
            for idx_k in range(self.n_known_outlier_classes):
                dist_unlabeled[:, idx_k+1] = torch.sum(
                    (outputs[is_unlabeled] - self.centroids[f'c_outlier_{idx_k+1}']) ** 2, dim=1)
            min_dist, _ = torch.min(dist_unlabeled, dim=1)
            loss_unlabeled = min_dist.mean()
        else:
            loss_unlabeled = torch.tensor(0.0, device=self.device)

        loss_cluster = loss_normal_cl + loss_outlier + loss_unlabeled

        loss = (self.coeff['sad']     * loss_sad
              + self.coeff['pred']    * loss_pred
              + self.coeff['dir']     * loss_dir
              + self.coeff['cluster'] * loss_cluster)

        return loss, loss_sad, loss_pred, loss_dir, loss_cluster

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, dataset: BaseADDataset, net: BaseNet, model_path=None, save=False):
        logger = logging.getLogger()

        train_loader, val_loader, _ = dataset.loaders(batch_size=self.batch_size,
                                                       num_workers=self.n_jobs_dataloader)
        logger.info(f'Training set size: {len(train_loader)}')
        logger.info(f'Validation set size: {len(val_loader)}')

        net = net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        logger.info('Initializing centers...')
        self.update_center(train_loader, net)
        logger.info('Centers initialized.')

        if self.aug_mode in ('nngmix', 'both'):
            self._build_nngmix_pool(train_loader)

        logger.info('Starting training physically...')
        start_time = time.time()
        best_auc = -np.inf
        net_store = None
        centroids_store = None
        per_class_thresholds_store = None

        for epoch in range(self.n_epochs):
            net.train()
            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = epoch_loss_pred = epoch_loss_sad = epoch_loss_dir = epoch_loss_cluster = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            for data in train_loader:
                inputs, _, semi_targets, _, signal_next = data
                inputs       = inputs.to(self.device)
                semi_targets = semi_targets.to(self.device)
                signal_next  = signal_next.to(self.device)

                inputs, semi_targets, signal_next = self.data_augmentation(inputs, semi_targets, signal_next)

                outputs, signal_pred = net(inputs)
                loss, loss_sad, loss_pred, loss_dir, loss_cluster = self.loss_all(
                    outputs, semi_targets, signal_pred, signal_next)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss         += loss.item()
                epoch_loss_pred    += loss_pred.item()
                epoch_loss_sad     += loss_sad.item()
                epoch_loss_dir     += loss_dir.item()
                epoch_loss_cluster += loss_cluster.item()
                n_batches += 1

            # Update centroids each epoch
            with torch.no_grad():
                self.update_center(train_loader, net)

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')
            wandb.log({'Train loss': epoch_loss / n_batches, 'Train time': epoch_train_time,
                       'loss_pred': epoch_loss_pred / n_batches, 'loss_sad': epoch_loss_sad / n_batches,
                       'loss_dir': epoch_loss_dir / n_batches, 'loss_cluster': epoch_loss_cluster / n_batches})

            # Validation
            if epoch % 100 == 0 and epoch > 0:
                val_auc, roc_curve_val, pct_val = self.val(val_loader, net)
                wandb.log({'Validation AUC': val_auc})
                if save:
                    torch.save({'centroids': self.centroids.copy(),
                                'roc': self.roc_curve,
                                'net_dict': net.state_dict().copy()},
                               model_path + f'/model_physical_{epoch}.tar')
                if val_auc > best_auc:
                    logger.info("Find better model.")
                    self.roc_curve             = roc_curve_val
                    best_auc                   = val_auc
                    net_store                  = copy.deepcopy(net)
                    centroids_store            = copy.deepcopy(self.centroids)
                    per_class_thresholds_store = pct_val

        self.centroids            = centroids_store
        self.per_class_thresholds = per_class_thresholds_store
        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')
        return net_store, best_auc

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------
    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        _, _, test_loader = dataset.loaders(batch_size=self.batch_size,
                                             num_workers=self.n_jobs_dataloader)
        logger.info(f'Test set size: {len(test_loader)}')
        net = net.to(self.device)

        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches  = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx, signal_next = data

                inputs       = inputs.to(self.device)
                labels       = labels.to(self.device)          # (batch, n_ac) multi-hot
                semi_targets = semi_targets.to(self.device)
                idx          = idx.to(self.device)
                signal_next  = signal_next.to(self.device)

                outputs, signal_pred = net(inputs)
                dist_norm = torch.sum((outputs - self.centroids['c_normal']) ** 2, dim=1)
                loss, loss_sad, loss_pred, loss_dir, loss_cluster = self.loss_all(
                    outputs, semi_targets, signal_pred, signal_next)

                scores = dist_norm

                idx_label_score += list(zip(
                    idx.cpu().numpy().tolist(),
                    labels.cpu().numpy().tolist(),          # list of multi-hot vectors
                    scores.cpu().numpy().tolist(),
                    outputs.cpu().numpy().tolist()
                ))

                epoch_loss += loss.item()
                n_batches  += 1

        self.test_time   = time.time() - start_time
        self.test_scores = idx_label_score

        # Unpack
        _, labels_list, scores_list, outputs_list = zip(*idx_label_score)

        # labels_list: list of (n_anomaly_classes,) vectors → stack to (n, n_ac)
        labels_arr  = np.stack(labels_list)                          # (n, n_ac)
        labels_bin  = (labels_arr.sum(axis=1) > 0).astype(int)      # binary: any anomaly
        scores_arr  = np.array(scores_list)
        outputs_arr = np.array(outputs_list)

        self.test_auc = roc_auc_score(labels_bin, scores_arr)

        # Best threshold from validation ROC curve
        fpr, tpr, thresholds = self.roc_curve
        youden_index  = tpr - fpr
        best_threshold = thresholds[np.argmax(youden_index)]

        samples_anomaly = scores_arr > best_threshold

        # ------------------------------------------------------------------
        # Multi-hot prediction via per-class distance thresholds
        # ------------------------------------------------------------------
        y_pred_mh = np.zeros((len(labels_arr), self.n_known_outlier_classes), dtype=int)

        if self.n_known_outlier_classes > 0 and self.per_class_thresholds is not None:
            centroid_stack = torch.stack([
                self.centroids[f'c_outlier_{i+1}']
                for i in range(self.n_known_outlier_classes)
            ]).to(self.device)
            outputs_tensor = torch.tensor(outputs_arr, device=self.device)
            dist_class     = torch.sum(
                (outputs_tensor.unsqueeze(1) - centroid_stack.unsqueeze(0)) ** 2, dim=2
            ).cpu().numpy()                                       # (n, n_known_ac)
            for i in range(self.n_known_outlier_classes):
                class_scores     = -dist_class[:, i]             # closer → higher score
                y_pred_mh[:, i]  = (class_scores > self.per_class_thresholds[i]).astype(int)

        # Gate: samples predicted normal get all-zero multi-hot
        y_pred_mh[~samples_anomaly] = 0

        # Unknown anomaly: anomalous but no known class fires
        y_pred_unknown = (samples_anomaly & (y_pred_mh.sum(axis=1) == 0)).astype(int)
        y_pred_full    = np.concatenate([y_pred_mh, y_pred_unknown[:, None]], axis=1)

        # ------------------------------------------------------------------
        # Ground-truth in the same (n_known_ac + 1) format
        # ------------------------------------------------------------------
        labels_mh      = labels_arr[:, self.known_col_indices].astype(int)  # (n, n_known_ac)
        labels_any     = (labels_arr.sum(axis=1) > 0)
        labels_unknown = (labels_any & (labels_mh.sum(axis=1) == 0)).astype(int)
        labels_full    = np.concatenate([labels_mh, labels_unknown[:, None]], axis=1)

        # Multi-label metrics
        self.test_f1_macro_mh = f1_score(labels_full, y_pred_full, average='macro',  zero_division=0)
        self.test_f1_micro_mh = f1_score(labels_full, y_pred_full, average='micro',  zero_division=0)
        self.test_hamming_acc = 1.0 - hamming_loss(labels_full, y_pred_full)
        self.test_subset_acc  = float(np.mean(np.all(labels_full == y_pred_full, axis=1)))

        # Binary anomaly-detection metrics (unchanged)
        y_pred_bin = samples_anomaly.astype(int)
        self.test_f1_binary        = f1_score(labels_bin, y_pred_bin, average='binary', zero_division=0)
        self.test_precision_binary = precision_score(labels_bin, y_pred_bin, zero_division=0)
        self.test_recall_binary    = recall_score(labels_bin, y_pred_bin, zero_division=0)
        self.test_acc_binary       = float(np.mean(labels_bin == y_pred_bin))

        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Multi-hot F1 macro:  {:.4f}'.format(self.test_f1_macro_mh))
        logger.info('Multi-hot F1 micro:  {:.4f}'.format(self.test_f1_micro_mh))
        logger.info('Multi-hot Hamming accuracy: {:.4f}'.format(self.test_hamming_acc))
        logger.info('Multi-hot Subset accuracy:  {:.4f}'.format(self.test_subset_acc))
        logger.info('Binary F1:        {:.4f}'.format(self.test_f1_binary))
        logger.info('Binary Precision: {:.4f}'.format(self.test_precision_binary))
        logger.info('Binary Recall:    {:.4f}'.format(self.test_recall_binary))
        logger.info('Binary Accuracy:  {:.4f}'.format(self.test_acc_binary))
        logger.info('Finished testing.')

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def val(self, val_loader, net: BaseNet):
        logger = logging.getLogger()

        logger.info('Starting validation...')
        epoch_loss = 0.0
        n_batches  = 0
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, labels, semi_targets, idx, signal_next = data

                inputs       = inputs.to(self.device)
                labels       = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx          = idx.to(self.device)
                signal_next  = signal_next.to(self.device)

                outputs, signal_pred = net(inputs)
                dist  = torch.sum((outputs - self.centroids['c_normal']) ** 2, dim=1)
                loss, loss_sad, loss_pred, loss_dir, loss_cluster = self.loss_all(
                    outputs, semi_targets, signal_pred, signal_next)

                scores = dist

                idx_label_score += list(zip(
                    idx.cpu().numpy().tolist(),
                    labels.cpu().numpy().tolist(),
                    scores.cpu().numpy().tolist(),
                    outputs.cpu().numpy().tolist()
                ))

                epoch_loss += loss.item()
                n_batches  += 1

        _, labels_list, scores_list, outputs_list = zip(*idx_label_score)
        labels_arr  = np.stack(labels_list)                       # (n, n_ac)
        labels_bin  = (labels_arr.sum(axis=1) > 0).astype(int)
        scores_arr  = np.array(scores_list)
        outputs_arr = np.array(outputs_list)                      # (n, rep_dim)

        val_auc          = roc_auc_score(labels_bin, scores_arr)
        fpr, tpr, thresholds = roc_curve(labels_bin, scores_arr, pos_label=1)

        # Per-class thresholds via Youden's index on each known anomaly class
        per_class_thresholds = []
        if self.n_known_outlier_classes > 0:
            centroid_stack = torch.stack([
                self.centroids[f'c_outlier_{i+1}']
                for i in range(self.n_known_outlier_classes)
            ]).cpu()                                              # (n_known_ac, rep_dim)
            outputs_t  = torch.tensor(outputs_arr)
            dist_class = torch.sum(
                (outputs_t.unsqueeze(1) - centroid_stack.unsqueeze(0)) ** 2, dim=2
            ).numpy()                                             # (n, n_known_ac)
            for i, col in enumerate(self.known_col_indices):
                gt           = labels_arr[:, col].astype(int)
                class_scores = -dist_class[:, i]                 # closer → higher score
                if gt.sum() > 0 and (gt == 0).sum() > 0:
                    fpr_c, tpr_c, thr_c = roc_curve(gt, class_scores, pos_label=1)
                    youden_c             = tpr_c - fpr_c
                    per_class_thresholds.append(float(thr_c[np.argmax(youden_c)]))
                else:
                    per_class_thresholds.append(float(-np.inf))

        logger.info('Val Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Val AUC: {:.2f}%'.format(100. * val_auc))
        logger.info('Finished validation.')
        return val_auc, (fpr, tpr, thresholds), per_class_thresholds

    # ------------------------------------------------------------------
    # Center update
    # ------------------------------------------------------------------
    def update_center(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Recompute hypersphere centers as means of labeled samples."""
        n_classes  = self.n_known_outlier_classes + 1
        n_samples  = torch.zeros(n_classes, device=self.device)

        self.centroids['c_normal'].zero_()
        for i in range(self.n_known_outlier_classes):
            self.centroids[f'c_outlier_{i+1}'].zero_()

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                inputs, target, semi_target, index, data_next = data
                # semi_target: (batch, n_anomaly_classes)
                semi_target = semi_target.to(self.device)
                inputs      = inputs.to(self.device)

                is_labeled_normal, _, _ = self._labeled_masks(semi_target)

                # Normal centroid
                if is_labeled_normal.any():
                    out_norm, _ = net(inputs[is_labeled_normal])
                    self.centroids['c_normal'] += out_norm.sum(dim=0)
                    n_samples[0]               += out_norm.shape[0]

                # Per-known-class centroids
                for idx_k, col in enumerate(self.known_col_indices):
                    mask_k = (semi_target[:, col] == 1)
                    if mask_k.any():
                        out_k, _ = net(inputs[mask_k])
                        self.centroids[f'c_outlier_{idx_k+1}'] += out_k.sum(dim=0)
                        n_samples[idx_k+1]                     += out_k.shape[0]

        if any(n_samples == 0):
            raise ValueError("At least one sample needs to be labeled in the training set for each class.")

        # Normalise and clip near-zero values away from 0
        for idx_k in range(n_classes):
            if idx_k == 0:
                c = self.centroids['c_normal']
            else:
                c = self.centroids[f'c_outlier_{idx_k}']
            c /= n_samples[idx_k]
            c[(c.abs() < eps) & (c < 0)] = -eps
            c[(c.abs() < eps) & (c > 0)] =  eps

    # ------------------------------------------------------------------
    # Data augmentation
    # ------------------------------------------------------------------
    def data_augmentation(self, inputs, semi_targets, signal_next):
        if self.aug_mode == 'gaussian':
            return self._gaussian_augmentation(inputs, semi_targets, signal_next)
        elif self.aug_mode == 'nngmix':
            return self._nngmix_augmentation(inputs, semi_targets, signal_next)
        elif self.aug_mode == 'both':
            inputs, semi_targets, signal_next = self._gaussian_augmentation(inputs, semi_targets, signal_next)
            return self._nngmix_augmentation(inputs, semi_targets, signal_next)
        else:
            raise ValueError(f"Unknown aug_mode: {self.aug_mode!r}")

    def _gaussian_augmentation(self, inputs, semi_targets, signal_next):
        is_labeled = ~(semi_targets == -1).all(dim=1)
        inputs_aug_list       = [inputs]
        semi_targets_aug_list = [semi_targets]
        signal_next_aug_list  = [signal_next]

        for _ in range(self.n_aug):
            noise       = torch.randn_like(inputs[is_labeled]) * 0.05
            inputs_aug  = inputs[is_labeled].clone() + noise
            st_aug      = semi_targets[is_labeled].clone()
            sn_aug      = signal_next[is_labeled].clone() + torch.randn_like(signal_next[is_labeled]) * 0.05
            inputs_aug_list.append(inputs_aug)
            semi_targets_aug_list.append(st_aug)
            signal_next_aug_list.append(sn_aug)

        return (torch.cat(inputs_aug_list, dim=0),
                torch.cat(semi_targets_aug_list, dim=0),
                torch.cat(signal_next_aug_list, dim=0))

    def _nngmix_augmentation(self, inputs, semi_targets, signal_next):
        """Appends one NNGMix pseudo-anomaly per labeled-anomaly sample in the batch.

        Mixes both the feature vector and signal_next with the same λ so that
        physics consistency is preserved across the augmented pair.
        """
        if self._nng_X_anomaly is None:
            return inputs, semi_targets, signal_next

        _, is_anom, _ = self._labeled_masks(semi_targets)
        if not is_anom.any():
            return inputs, semi_targets, signal_next

        cfg    = self._nng_cfg
        device = inputs.device

        n          = is_anom.sum().item()
        X_anom_b   = inputs[is_anom].cpu().numpy().reshape(n, -1).astype(np.float32)
        sn_anom_b  = signal_next[is_anom].cpu().numpy().reshape(n, -1).astype(np.float32)
        st_anom_b  = semi_targets[is_anom]

        d    = X_anom_b.shape[1]
        d_sn = sn_anom_b.shape[1]
        k_norm = max(1, min(cfg['nn_k'],         len(self._nng_X_pool)))
        k_anom = max(1, min(cfg['nn_k_anomaly'], len(self._nng_X_anomaly)))

        X_pseudo  = np.empty((n, d),    dtype=np.float32)
        sn_pseudo = np.empty((n, d_sn), dtype=np.float32)

        for i in range(n):
            use_normal = (np.random.uniform() > 0.5) and (self._nng_tree_normal is not None)

            if use_normal:
                _, ind = self._nng_tree_normal.query(X_anom_b[i:i+1], k=k_norm)
                i2 = int(np.random.choice(ind[0]))
                x_p  = self._nng_X_pool[i2]
                sn_p = self._nng_sn_pool[i2]
            elif self._nng_tree_anomaly is not None:
                _, ind = self._nng_tree_anomaly.query(X_anom_b[i:i+1], k=k_anom)
                i2 = int(np.random.choice(ind[0]))
                x_p  = self._nng_X_anomaly[i2]
                sn_p = self._nng_sn_anomaly[i2]
            else:
                x_p, sn_p = X_anom_b[i], sn_anom_b[i]

            lam = (np.random.uniform() if cfg['use_uniform']
                   else float(np.random.beta(cfg['mixup_alpha'], cfg['mixup_beta'])))

            if cfg['nn_mix_gaussian']:
                std = cfg['nn_mix_gaussian_std']
                x_a = X_anom_b[i] + np.random.normal(0, std, d).astype(np.float32)
                x_p = x_p         + np.random.normal(0, std, d).astype(np.float32)
            else:
                x_a = X_anom_b[i]

            X_pseudo[i] = lam * x_a + (1 - lam) * x_p
            sn_pseudo[i] = lam * sn_anom_b[i] + (1 - lam) * sn_p

        X_t  = torch.tensor(X_pseudo,  dtype=inputs.dtype,       device=device).view(-1, *inputs.shape[1:])
        sn_t = torch.tensor(sn_pseudo, dtype=signal_next.dtype,  device=device).view(-1, *signal_next.shape[1:])

        return (
            torch.cat([inputs,       X_t],      dim=0),
            torch.cat([semi_targets, st_anom_b], dim=0),
            torch.cat([signal_next,  sn_t],     dim=0),
        )

    def _build_nngmix_pool(self, train_loader: DataLoader):
        """One-time pass over the training set to build KD-trees for NNGMix."""
        from scipy import spatial as sp_spatial
        logger = logging.getLogger()

        X_anom_list,  sn_anom_list = [], []
        X_pool_list,  sn_pool_list = [], []

        for data in train_loader:
            inputs, _, semi_targets, _, signal_next = data
            X  = inputs.numpy().reshape(len(inputs), -1).astype(np.float32)
            SN = signal_next.numpy().reshape(len(inputs), -1).astype(np.float32)
            ST = semi_targets.numpy()

            is_unlabeled       = (ST == -1).all(axis=1)
            is_labeled_normal  = (~is_unlabeled) & (ST == 0).all(axis=1)
            is_labeled_anomaly = (~is_unlabeled) & (ST > 0).any(axis=1)

            if is_labeled_anomaly.any():
                X_anom_list.append(X[is_labeled_anomaly])
                sn_anom_list.append(SN[is_labeled_anomaly])

            pool_mask = is_unlabeled | is_labeled_normal
            if pool_mask.any():
                X_pool_list.append(X[pool_mask])
                sn_pool_list.append(SN[pool_mask])

        if not X_anom_list:
            logger.warning('NNGMix: no labeled anomaly samples found; NNGMix augmentation disabled.')
            return

        self._nng_X_anomaly  = np.vstack(X_anom_list)
        self._nng_sn_anomaly = np.vstack(sn_anom_list)
        feat_dim = self._nng_X_anomaly.shape[1]
        sn_dim   = self._nng_sn_anomaly.shape[1]
        self._nng_X_pool  = np.vstack(X_pool_list)  if X_pool_list  else np.empty((0, feat_dim),  dtype=np.float32)
        self._nng_sn_pool = np.vstack(sn_pool_list) if sn_pool_list else np.empty((0, sn_dim),    dtype=np.float32)

        self._nng_tree_normal  = sp_spatial.KDTree(self._nng_X_pool)    if len(self._nng_X_pool)    > 0 else None
        self._nng_tree_anomaly = sp_spatial.KDTree(self._nng_X_anomaly) if len(self._nng_X_anomaly) > 1 else None

        logger.info(f'NNGMix pool built: {len(self._nng_X_anomaly)} anomaly, '
                    f'{len(self._nng_X_pool)} normal-pool samples.')
