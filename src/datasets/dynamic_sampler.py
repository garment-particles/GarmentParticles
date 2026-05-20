import torch
from torch.utils.data import Sampler
import torch.distributed as dist
import numpy as np
from typing import List, Optional
import pickle
import os
from collections import defaultdict

class ConstantTokenBatchSampler(Sampler):
    """
    Creates batches with approximately constant total token count.
    Dynamically adjusts batch size to maintain constant computational load.
    """
    def __init__(
        self,
        dataset,
        target_tokens_per_batch: int,  # Tune based on GPU memory
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        token_counts_cache: Optional[str] = None,
        max_batch_size: int = 128,  # Safety limit
        min_batch_size: int = 1,
        bucket_size_bins: int = 10,  # Number of bins for grouping similar sizes
        drop_last: bool = False,
        verbose: bool = True,
    ):
        self.num_replicas = num_replicas
        self.rank = rank
        self.dataset = dataset
        self.target_tokens_per_batch = target_tokens_per_batch
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.bucket_size_bins = bucket_size_bins
        self.drop_last = drop_last
        self.verbose = verbose and rank == 0
        
        # Load or compute token counts
        if token_counts_cache and os.path.exists(token_counts_cache):
            if self.verbose:
                print(f"Loading token counts from {token_counts_cache}")
            with open(token_counts_cache, 'rb') as f:
                self.token_counts = pickle.load(f)
        else:
            if self.verbose:
                print("Computing token counts...")
            self.token_counts = self._compute_token_counts()
            if token_counts_cache and rank == 0:
                with open(token_counts_cache, 'wb') as f:
                    pickle.dump(self.token_counts, f)
                if self.verbose:
                    print(f"Saved token counts to {token_counts_cache}")
        
        # Assign samples to ranks for token balancing
        self.rank_assignments = self._assign_samples_to_ranks()
        
        # Create batches for this rank
        self.batches = self._create_constant_token_batches()
        
        self._print_statistics()
    
    def _compute_token_counts(self) -> List[int]:
        """Compute number of tokens (points) for each sample."""
        token_counts = []
        for idx in range(len(self.dataset)):
            file = self.dataset.files[idx]
            stats_file = os.path.join(os.path.dirname(file), 'stats.txt')
            if os.path.exists(stats_file):
                stats = np.loadtxt(stats_file)
                token_counts.append(int(stats[0]))
            elif file.endswith('.npz'):
                data = np.load(file)
                token_counts.append(int(data['pointcloud'].shape[0]))
            else:
                raise FileNotFoundError(
                    f"Cannot determine token count for {file}: "
                    f"no stats.txt found and file is not .npz"
                )

            if (idx + 1) % 1000 == 0 and self.verbose:
                print(f"  Processed {idx + 1}/{len(self.dataset)} files "
                      f"({(idx+1)/len(self.dataset)*100:.1f}%)")

        return token_counts
    
    def _assign_samples_to_ranks(self) -> List[List[int]]:
        """
        Assign samples to ranks to balance total token counts.
        Uses greedy bin-packing algorithm.
        """
        # Create list of (index, token_count) tuples
        samples = [(idx, count) for idx, count in enumerate(self.token_counts)]
        
        # Sort by token count descending (largest first for better balance)
        samples.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize assignments
        rank_assignments = [[] for _ in range(self.num_replicas)]
        rank_token_totals = [0] * self.num_replicas
        
        # Greedy assignment
        for idx, token_count in samples:
            # Assign to rank with minimum current total
            min_rank = min(range(self.num_replicas), key=lambda r: rank_token_totals[r])
            rank_assignments[min_rank].append(idx)
            rank_token_totals[min_rank] += token_count
        
        return rank_assignments
    
    def _create_constant_token_batches(self) -> List[List[int]]:
        """
        Create batches with approximately constant token count.
        Groups similar-sized samples together for efficiency.
        """
        indices = self.rank_assignments[self.rank]
        
        # Create bins based on token count
        # This groups similar-sized samples together
        min_tokens = min(self.token_counts[idx] for idx in indices)
        max_tokens = max(self.token_counts[idx] for idx in indices)
        
        # Create bins
        bin_edges = np.linspace(min_tokens, max_tokens, self.bucket_size_bins + 1)
        bins = defaultdict(list)
        
        for idx in indices:
            token_count = self.token_counts[idx]
            # Find which bin this belongs to
            bin_idx = np.digitize(token_count, bin_edges) - 1
            bin_idx = max(0, min(bin_idx, self.bucket_size_bins - 1))
            bins[bin_idx].append(idx)
        
        # Create batches from each bin
        batches = []
        for bin_idx in sorted(bins.keys()):
            bin_indices = bins[bin_idx]
            
            # Shuffle within bin if needed
            if self.shuffle:
                np.random.seed(self.seed + self.epoch + bin_idx)
                np.random.shuffle(bin_indices)
            
            # Create batches with constant token count
            current_batch = []
            current_tokens = 0
            
            for idx in bin_indices:
                token_count = self.token_counts[idx]
                
                # Check if we should start a new batch
                would_exceed = current_tokens + token_count > self.target_tokens_per_batch * 1.2
                at_max_size = len(current_batch) >= self.max_batch_size
                
                if current_batch and (would_exceed or at_max_size):
                    # Save current batch if it meets minimum size
                    if len(current_batch) >= self.min_batch_size:
                        batches.append(current_batch)
                    
                    # Start new batch
                    current_batch = [idx]
                    current_tokens = token_count
                else:
                    # Add to current batch
                    current_batch.append(idx)
                    current_tokens += token_count
            
            # Add last batch if it meets minimum size
            if len(current_batch) >= self.min_batch_size:
                batches.append(current_batch)
            elif not self.drop_last and current_batch:
                # Include even if smaller than minimum
                batches.append(current_batch)
        
        return batches
    
    def _print_statistics(self):
        """Print detailed statistics about batch distribution."""
        # Compute statistics
        batch_sizes = [len(batch) for batch in self.batches]
        batch_tokens = [
            sum(self.token_counts[idx] for idx in batch)
            for batch in self.batches
        ]
        
        # Gather stats from all ranks
        all_batch_sizes = [None] * self.num_replicas
        all_batch_tokens = [None] * self.num_replicas
        all_num_batches = [None] * self.num_replicas
        
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(all_batch_sizes, batch_sizes)
            dist.all_gather_object(all_batch_tokens, batch_tokens)
            dist.all_gather_object(all_num_batches, len(self.batches))
        else:
            all_batch_sizes[0] = batch_sizes
            all_batch_tokens[0] = batch_tokens
            all_num_batches[0] = len(self.batches)
        
        if self.rank == 0:
            print("\n" + "="*80)
            print("Constant Token Batch Sampler Statistics")
            print("="*80)
            
            # Per-rank statistics
            rank_token_totals = []
            for r in range(self.num_replicas):
                total_tokens = sum(self.token_counts[idx] 
                                 for idx in self.rank_assignments[r])
                num_samples = len(self.rank_assignments[r])
                num_batches = all_num_batches[r]
                rank_token_totals.append(total_tokens)
                
                print(f"\nRank {r}:")
                print(f"  Total samples: {num_samples:,}")
                print(f"  Total tokens: {total_tokens:,}")
                print(f"  Number of batches: {num_batches}")
                
                if all_batch_sizes[r]:
                    print(f"  Batch size: min={min(all_batch_sizes[r])}, "
                          f"max={max(all_batch_sizes[r])}, "
                          f"avg={np.mean(all_batch_sizes[r]):.1f}")
                    print(f"  Tokens/batch: min={min(all_batch_tokens[r]):,}, "
                          f"max={max(all_batch_tokens[r]):,}, "
                          f"avg={np.mean(all_batch_tokens[r]):,.0f} "
                          f"(target: {self.target_tokens_per_batch:,})")
            
            # Overall balance
            print(f"\nOverall Load Balance:")
            avg_tokens = np.mean(rank_token_totals)
            max_tokens = max(rank_token_totals)
            min_tokens = min(rank_token_totals)
            imbalance = (max_tokens - min_tokens) / avg_tokens * 100
            
            print(f"  Avg tokens/rank: {avg_tokens:,.0f}")
            print(f"  Token imbalance: {imbalance:.2f}% (max-min)")
            print(f"  Token std dev: {np.std(rank_token_totals):,.0f}")
            print("="*80 + "\n")
    
    def __iter__(self):
        """Yield batches for this epoch."""
        # Recreate batches with new shuffling
        if self.shuffle:
            self.batches = self._create_constant_token_batches()
            
            # Shuffle batch order
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_order = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in batch_order]
        else:
            batches = self.batches
        
        # Yield entire batches (not individual indices)
        for batch in batches:
            yield batch
    
    def __len__(self):
        """Return number of batches."""
        return len(self.batches)
    
    def set_epoch(self, epoch: int):
        """Set epoch for proper shuffling."""
        self.epoch = epoch


class ConstantTokenCollator:
    """
    Collator for variable-sized batches from ConstantTokenBatchSampler.
    Pads to max length within each batch.
    """
    def __init__(self):
        pass
    
    def __call__(self, batch):
        """Collate variable-sized batch."""
        # Find max point cloud length in this batch
        pcd_lens = [item['input'].shape[0] for item in batch]
        max_pcd_len = max(pcd_lens)
        cu_pcd_lens = torch.cumsum(torch.tensor(pcd_lens), dim=0)
        cu_pcd_lens = torch.cat([torch.zeros(1), cu_pcd_lens])
        
        pcds = torch.cat([item['input'] for item in batch], dim=0)       
        batch_dict = {
            'input': pcds,
            'cu_input_lens': cu_pcd_lens.int(),
            'max_input_len': max_pcd_len,
            "batch_size": len(batch),
        }
        if 'text_tokens' in batch[0]:
            text_tokens = torch.stack([item['text_tokens'] for item in batch], dim=0)
            text_attn_mask = torch.stack([item['text_attn_mask'] for item in batch], dim=0)
            batch_dict.update({
                'text_tokens': text_tokens,
                'text_attn_mask': text_attn_mask.bool(),
            })
        if 'pixel_values' in batch[0]:
            pixel_values = torch.stack([item['pixel_values'] for item in batch], dim=0)
            image_paths = [item['image_path'] for item in batch]
            batch_dict.update({
                'pixel_values': pixel_values,
                'image_path': image_paths,
            })
        
        return batch_dict