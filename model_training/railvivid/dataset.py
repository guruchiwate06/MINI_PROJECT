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
        self.partners={}
        if augment and cfg['training']['cross_run_weight']>0:
            from .geometry import nearest_indices
            byrun={r:g.sort_values('chainage_m') for r,g in self.frame.groupby('run_id')}
            for i,row in self.frame.iterrows():
                matches=[]
                for run,g in byrun.items():
                    if run==row.run_id:continue
                    j=nearest_indices(g.chainage_m.to_numpy(),np.asarray([row.chainage_m]))[0];other=g.iloc[j]
                    if abs(other.chainage_m-row.chainage_m)<=cfg['training']['cross_run_tolerance_m']:matches.append(int(other.name))
                if matches:self.partners[i]=matches
        self.k=len(cfg['alignment']['candidate_offsets_m']);self.dim=cfg['model']['vision_dim']
        if vision:
            required={f for ids in frame.candidate_frame_ids for f in ids}
            missing=required-set(vision.lookup)
            if missing:raise ArtifactError(f'Vision cache lacks {len(missing)} required frames. Extract the matching cache before training.')

    def __getstate__(self):
        state=self.__dict__.copy();state['_arrays']=OrderedDict();return state

    def __len__(self):return len(self.frame)

    def array(self,run):
        if run not in self._arrays:
            path=self.root/'runs'/run/'acceleration.npy'
            try:self._arrays[run]=np.load(path,mmap_mode='r')
            except (OSError,ValueError) as e:raise ArtifactError(f'Cannot open waveform cache {path}: {e}') from e
            if len(self._arrays)>4:self._arrays.popitem(last=False)
        self._arrays.move_to_end(run);return self._arrays[run]

    def __getitem__(self,index):
        row=self.frame.iloc[index]
        raw=self.array(row.run_id)
        if row.start_idx<0 or row.end_idx>len(raw) or row.end_idx-row.start_idx!=self.cfg['data']['window_samples']:
            raise DataIntegrityError(f'{row.window_id}: raw source interval is out of bounds')
        wave=np.asarray(raw[int(row.start_idx):int(row.end_idx),self.channels],dtype='float32').T.copy()
        finite(wave,row.window_id)
        wave=(wave-np.asarray(self.normalizer['mean'],dtype='float32')[:,None])/np.asarray(self.normalizer['std'],dtype='float32')[:,None]
        item_seed=int.from_bytes(hashlib.sha256(f"{self.cfg['training']['seed']}:{self.epoch}:{row.window_id}".encode()).digest()[:8],'little')
        rng=np.random.default_rng(item_seed)
        if self.augment:wave=wave*rng.uniform(.95,1.05)+rng.normal(0,.02,wave.shape).astype('float32')
        noise=float(self.corruption.get('noise_std_normalized',0))
        if noise:wave=wave+rng.normal(0,noise,wave.shape).astype('float32')
        for channel in self.corruption.get('missing_channels',[]):
            where=np.flatnonzero(self.channels==channel-1)
            if len(where):wave[where]=0
        if self.corruption.get('missing_vibration'):wave[:]=0
        vision=np.zeros((self.k,self.dim),dtype='float32');quality=np.zeros((self.k,2),dtype='float32')
        mask=np.zeros(self.k,dtype=bool);offsets=np.zeros(self.k,dtype='float32')
        if self.vision and not self.corruption.get('missing_vision'):
            for j,(fid,delta) in enumerate(zip(row.candidate_frame_ids,row.candidate_offsets_m)):
                if j>=self.k:raise DataIntegrityError('Candidate count exceeds configured capacity')
                p=self.vision.lookup[fid];vision[j]=self.vision.features[p];quality[j]=self.vision.quality[p];mask[j]=True;offsets[j]=delta
        if self.augment and self.cfg['model']['kind']=='fusion' and rng.random()<self.cfg['model']['modality_dropout']:mask[:]=False;vision[:]=0
        partner=np.zeros_like(wave);partner_available=False
        if index in self.partners:
            other=self.frame.iloc[self.partners[index][self.epoch%len(self.partners[index])]]
            partner=np.asarray(self.array(other.run_id)[int(other.start_idx):int(other.end_idx),self.channels],dtype='float32').T.copy()
            partner=(partner-np.asarray(self.normalizer['mean'],dtype='float32')[:,None])/np.asarray(self.normalizer['std'],dtype='float32')[:,None]
            partner_available=True
        offsets=offsets-float(self.corruption.get('alignment_prior_shift_m',0))
        finite(vision,'vision cache');finite(quality,'image quality')
        return {'waveform':torch.from_numpy(wave.astype('float32')),'vision':torch.from_numpy(vision),
            'vision_mask':torch.from_numpy(mask),'offsets':torch.from_numpy(offsets),'quality':torch.from_numpy(quality),
            'label':torch.tensor(float(row.label)),'index':torch.tensor(index),
            'partner_waveform':torch.from_numpy(partner),'partner_available':torch.tensor(partner_available)}

def collate_to_device(batch,device):return {k:v.to(device,non_blocking=True) for k,v in batch.items()}
