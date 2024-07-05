from typing import Dict, Union, List, Any, Tuple

import torch
from TTS.tts.datasets import TTSDataset
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer
from coqpit import Coqpit
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, BatchSampler


class AllTalkTrainer(GPTTrainer):
    def __init__(self, config: Coqpit):
        super().__init__(config)

    def get_sampler(self, dataset: TTSDataset, config: Coqpit, num_gpus=1):
        # sampler for DDP
        if num_gpus > 1:
            batch_sampler = DistributedSampler(dataset)
        else:
            print(f"[FINETUNE] creating training sampler with {len(dataset)} samples")
            random_sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))
            batch_sampler = BatchSampler(random_sampler, batch_size=self.batch_size, drop_last=True)
        return batch_sampler

    def get_data_loader(
            self,
            config: Coqpit,
            assets: Dict,
            is_eval: bool,
            samples: Union[List[Dict], List[List]],
            verbose: bool,
            num_gpus: int,
            rank: int = None,
    ) -> "DataLoader":  # pylint: disable=W0613
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = XTTSDataset(self.config, samples, self.xtts.tokenizer, config.audio.sample_rate, is_eval)

            # wait all the DDP process to be ready
            if num_gpus > 1:
                torch.distributed.barrier()

            # sort input sequences from short to long
            # dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(dataset, config, num_gpus)

            # ignore sampler when is eval because if we changed the sampler parameter we will not be able to compare previous runs
            if sampler is None or is_eval:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            elif not is_eval and isinstance(sampler, BatchSampler):
                print(f"[FINETUNE] Using custom batch sampler")
                loader = DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=True,
                )
            else:
                loader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
        return loader

    def optimize(self, *args: Any, **kwargs: Any) -> Tuple[Dict, Dict, float]:
        return super().optimize(args, kwargs)

    @staticmethod
    def init_from_config(config: "GPTTrainerConfig", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (GPTTrainerConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        return AllTalkTrainer(config)
