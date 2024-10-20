import torch
from torch.utils.data.dataset import ConcatDataset


class MyFirstDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat([torch.ones(1,2).repeat(5, 1), -torch.ones(1,2).repeat(5, 1)], dim=0) #torch.cat((-torch.ones(5), torch.ones(5)))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]


class MySecondDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat((torch.ones(50) * 5, torch.ones(5) * -5))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]
    
class MyThirdDataset(torch.utils.data.Dataset):
    def __init__(self):
        # dummy dataset
        self.samples = torch.cat([3*torch.ones(1,3).repeat(5, 1), -3*torch.ones(1,3).repeat(5, 1)], dim=0) #torch.cat((-torch.ones(5), torch.ones(5)))

    def __getitem__(self, index):
        # change this to your samples fetching logic
        return self.samples[index]

    def __len__(self):
        # change this to return number of samples in your dataset
        return self.samples.shape[0]


first_dataset = MyFirstDataset()
second_dataset = MySecondDataset()
third_dataset = MyThirdDataset()
concat_dataset = ConcatDataset([first_dataset, second_dataset, third_dataset])


import math
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size, sampler=SequentialSampler):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.samples) for cur_dataset in dataset.datasets])
        self.sampler = sampler

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = self.sampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)

batch_size = 8

# dataloader with BatchSchedulerSampler
dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                        sampler=BatchSchedulerSampler(dataset=concat_dataset,
                                                                    batch_size=batch_size,
                                                                #    sampler=SequentialSampler,
                                                                    sampler=RandomSampler,
                                                                    ),
                                        batch_size=batch_size,
                                        shuffle=False)

for inputs in dataloader:
    print(inputs)