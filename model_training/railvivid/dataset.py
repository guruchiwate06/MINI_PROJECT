from __future__ import annotations
import hashlib,json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import finite,read_json,fingerprint,write_json,file_hash
from .errors import ArtifactError,DataIntegrityError
from .prepare import load_prepared

def merged_intervals(frame):
    result=[]
    for start,end in frame[['start_idx','end_idx']].sort_values('start_idx').itertuples(index=False,name=None):
        if result and start<=result[-1][1]:result[-1][1]=max(result[-1][1],int(end))
        else:result.append([int(start),int(end)])
    return result

def training_normalizer(root,train,channels):
    """Count unique raw training samples once, never validation/test/run-wide data."""
    count=0;total=np.zeros(len(channels));squares=np.zeros(len(channels));ix=np.asarray(channels)-1
    for run,g in train.groupby('run_id'):
        raw=np.load(Path(root)/'runs'/run/'acceleration.npy',mmap_mode='r')
        for start,end in merged_intervals(g):
            for offset in range(start,end,100000):
                a=np.asarray(raw[offset:min(offset+100000,end),ix],dtype='float64');finite(a,'training normalization')
                total+=a.sum(0);squares+=(a*a).sum(0);count+=len(a)
    if count<2:raise DataIntegrityError('Insufficient training samples for normalization')
    mean=total/count;std=np.sqrt(np.maximum(squares/count-mean**2,0))
    if (std<1e-8).any():raise DataIntegrityError('A selected training channel has zero/near-zero variance')
    return {'mean':mean.tolist(),'std':std.tolist(),'channels':channels,'unique_training_samples':count,
            'source_window_hash':fingerprint(sorted(train.window_id.tolist()))}

class VisionStore:
    def __init__(self,folder,expected_dim=None):
        import pandas as pd
        self.folder=Path(folder);self.meta=read_json(self.folder/'READY.json')
        for name in ['frames.csv','embeddings.npy','quality.npy']:
            if name not in self.meta.get('artifacts',{}) or file_hash(self.folder/name)!=self.meta['artifacts'][name]:
                raise DataIntegrityError(f'Vision cache file absent or modified: {name}')
        self.index=pd.read_csv(self.folder/'frames.csv');self.lookup=dict(zip(self.index.frame_id,range(len(self.index))))
        if len(self.lookup)!=len(self.index):raise DataIntegrityError('Duplicate frame IDs in vision cache')
        self.features=np.load(self.folder/'embeddings.npy',mmap_mode='r')
        self.quality=np.load(self.folder/'quality.npy',mmap_mode='r')
        if len(self.features)!=len(self.index) or len(self.quality)!=len(self.index):raise DataIntegrityError('Vision cache/index length mismatch')
        if expected_dim and self.features.shape[1]!=expected_dim:raise ConfigurationErrorForVision(f'Vision embedding dimension {self.features.shape[1]} != configured {expected_dim}')

    def __getstate__(self):return {'folder':self.folder}
    def __setstate__(self,state):self.__init__(state['folder'])

from .errors import ConfigurationError as ConfigurationErrorForVision

class WindowDataset(Dataset):
    def __init__(self,root,frame,normalizer,cfg,vision=None,augment=False,corruption=None):
        self.root=Path(root);self.frame=frame.reset_index(drop=True);self.normalizer=normalizer;self.cfg=cfg
        self.vision=vision;self.augment=augment;self.epoch=0;self.corruption=corruption or {};self._arrays=OrderedDict()
        self.channels=np.asarray(cfg['model']['channels'])-1
        # Compact metadata is prepared once; the epoch hot path never creates Pandas rows.
        self.run_ids=self.frame.run_id.to_numpy(copy=True)
        self.starts=self.frame.start_idx.to_numpy(dtype='int64',copy=True)
        self.ends=self.frame.end_idx.to_numpy(dtype='int64',copy=True)
        self.window_ids=self.frame.window_id.to_numpy(copy=True)
        self.labels=self.frame.label.to_numpy(dtype='float32',copy=True)
        self.mean=np.asarray(normalizer['mean'],dtype='float32')[:,None]
        self.std=np.asarray(normalizer['std'],dtype='float32')[:,None]
        self.max_open_runs=cfg['training']['max_open_runs']
        self.partners={}
        if augment and cfg['training']['cross_run_weight']>0:
            from .geometry import nearest_indices
            byrun={r:g.sort_values('chainage_m') for r,g in self.frame.groupby('run_id')}
            for run,g in byrun.items():
                for other_run,other in byrun.items():
                    if run==other_run:continue
                    distances=other.chainage_m.to_numpy();queries=g.chainage_m.to_numpy()
                    nearest=nearest_indices(distances,queries)
                    permitted=np.abs(distances[nearest]-queries)<=cfg['training']['cross_run_tolerance_m']
                    for i,j in zip(g.index.to_numpy()[permitted],other.index.to_numpy()[nearest[permitted]]):
                        self.partners.setdefault(int(i),[]).append(int(j))
        self.k=len(cfg['alignment']['candidate_offsets_m']);self.dim=cfg['model']['vision_dim']
        self.vision_indices=np.full((len(self.frame),self.k),-1,dtype='int64')
        self.offset_values=np.zeros((len(self.frame),self.k),dtype='float32')
        if vision:
            required={f for ids in frame.candidate_frame_ids for f in ids}
            missing=required-set(vision.lookup)
            if missing:raise ArtifactError(f'Vision cache lacks {len(missing)} required frames. Extract the matching cache before training.')
            for i,(ids,offsets) in enumerate(zip(self.frame.candidate_frame_ids,self.frame.candidate_offsets_m)):
                if len(ids)>self.k or len(ids)!=len(offsets):raise DataIntegrityError('Invalid vision candidate/offset capacity')
                self.vision_indices[i,:len(ids)]=[vision.lookup[fid] for fid in ids]
                self.offset_values[i,:len(ids)]=offsets

    def __getstate__(self):
        state=self.__dict__.copy();state['_arrays']=OrderedDict();return state

    def __len__(self):return len(self.frame)

    def array(self,run):
        if run not in self._arrays:
            path=self.root/'runs'/run/'acceleration.npy'
            try:self._arrays[run]=np.load(path,mmap_mode='r')
            except (OSError,ValueError) as e:raise ArtifactError(f'Cannot open waveform cache {path}: {e}') from e
            if len(self._arrays)>self.max_open_runs:self._arrays.popitem(last=False)
        self._arrays.move_to_end(run);return self._arrays[run]

    def __getitem__(self,index):
        raw=self.array(self.run_ids[index]);start=self.starts[index];end=self.ends[index]
        if start<0 or end>len(raw) or end-start!=self.cfg['data']['window_samples']:
            raise DataIntegrityError(f'{self.window_ids[index]}: raw source interval is out of bounds')
        wave=np.asarray(raw[start:end,self.channels],dtype='float32').T.copy()
        finite(wave,self.window_ids[index]);wave=(wave-self.mean)/self.std
        noise=float(self.corruption.get('noise_std_normalized',0))
        rng=None
        if self.augment or noise:
            item_seed=int.from_bytes(hashlib.sha256(f"{self.cfg['training']['seed']}:{self.epoch}:{self.window_ids[index]}".encode()).digest()[:8],'little')
            rng=np.random.default_rng(item_seed)
        if self.augment:wave=wave*rng.uniform(.95,1.05)+rng.normal(0,.02,wave.shape).astype('float32')
        if noise:wave=wave+rng.normal(0,noise,wave.shape).astype('float32')
        for channel in self.corruption.get('missing_channels',[]):
            where=np.flatnonzero(self.channels==channel-1)
            if len(where):wave[where]=0
        if self.corruption.get('missing_vibration'):wave[:]=0
        vision=np.zeros((self.k,self.dim),dtype='float32');quality=np.zeros((self.k,2),dtype='float32')
        mask=np.zeros(self.k,dtype=bool);offsets=np.zeros(self.k,dtype='float32')
        if self.vision and not self.corruption.get('missing_vision'):
            mask=self.vision_indices[index]>=0;positions=self.vision_indices[index,mask]
            vision[mask]=self.vision.features[positions];quality[mask]=self.vision.quality[positions]
            offsets=self.offset_values[index].copy()
        if self.augment and self.cfg['model']['kind']=='fusion' and rng.random()<self.cfg['model']['modality_dropout']:mask[:]=False;vision[:]=0
        partner=np.zeros_like(wave);partner_available=False
        if index in self.partners:
            other=self.partners[index][self.epoch%len(self.partners[index])]
            partner=np.asarray(self.array(self.run_ids[other])[self.starts[other]:self.ends[other],self.channels],dtype='float32').T.copy()
            finite(partner,self.window_ids[other]);partner=(partner-self.mean)/self.std
            partner_available=True
        offsets=offsets-float(self.corruption.get('alignment_prior_shift_m',0))
        finite(vision,'vision cache');finite(quality,'image quality')
        return {'waveform':torch.from_numpy(wave.astype('float32')),'vision':torch.from_numpy(vision),
            'vision_mask':torch.from_numpy(mask),'offsets':torch.from_numpy(offsets),'quality':torch.from_numpy(quality),
            'label':torch.tensor(float(self.labels[index])),'index':torch.tensor(index),
            'partner_waveform':torch.from_numpy(partner),'partner_available':torch.tensor(partner_available)}

def collate_to_device(batch,device):return {k:v.to(device,non_blocking=True) for k,v in batch.items()}
