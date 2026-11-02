"""Small models with explicit spatial candidates and safe missing-modality paths."""
import torch
from torch import nn
from src.models.cnn1d import Vibration1DCNN

class ResidualBlock(nn.Module):
    def __init__(self,width,kernel,dilation,dropout):
        super().__init__();padding=(kernel//2)*dilation
        self.body=nn.Sequential(nn.Conv1d(width,width,kernel,padding=padding,dilation=dilation),nn.GroupNorm(1,width),nn.GELU(),
            nn.Dropout(dropout),nn.Conv1d(width,width,kernel,padding=padding,dilation=dilation),nn.GroupNorm(1,width),nn.GELU())
    def forward(self,x):return x+self.body(x)

class TemporalEncoder(nn.Module):
    """Noncausal dilated TCN for complete-window classification, not forecasting."""
    def __init__(self,channels,width,depth,kernel,embedding,dropout):
        super().__init__()
        if kernel%2!=1:raise ValueError('TCN kernel must be odd to preserve residual lengths')
        self.net=nn.Sequential(nn.Conv1d(channels,width,15,stride=4,padding=7),nn.GroupNorm(1,width),nn.GELU(),
            *[ResidualBlock(width,kernel,2**i,dropout) for i in range(depth)],nn.Conv1d(width,embedding,1),nn.AdaptiveAvgPool1d(16))
    def tokens(self,x):return self.net(x).transpose(1,2)
    def forward(self,x):return self.tokens(x).mean(1)

class ResearchModel(nn.Module):
    def __init__(self,cfg):
        super().__init__();m=cfg['model'];self.kind=m['kind'];self.fusion=m['fusion'];d=m['embedding_dim']
        if self.kind=='cnn':
            self.cnn=Vibration1DCNN(in_channels=len(m['channels']));return
        if self.kind!='vision':self.encoder=TemporalEncoder(len(m['channels']),m['width'],m['depth'],m['kernel_size'],d,m['dropout'])
        if self.kind=='tcn':
            self.vibration_head=nn.Linear(d,1);return
        self.vision_projection=nn.Sequential(nn.LayerNorm(m['vision_dim']),nn.Linear(m['vision_dim'],d),nn.GELU())
        self.offset_projection=nn.Sequential(nn.Linear(1,d),nn.Tanh())
        self.attention=nn.MultiheadAttention(d,m['attention_heads'],dropout=m['dropout'],batch_first=True)
        self.gate=nn.Sequential(nn.Linear(2*d+3,d),nn.GELU(),nn.Linear(d,1),nn.Sigmoid())
        self.vibration_head=nn.Linear(d,1)
        self.vision_head=nn.Linear(d,1)
        self.head=nn.Sequential(nn.LayerNorm(2*d+1),nn.Linear(2*d+1,d),nn.GELU(),nn.Dropout(m['dropout']),nn.Linear(d,1))
        self.distance_scale=cfg['alignment']['pooling_bandwidth_m']

    def forward(self,batch):
        return self.forward_features(batch)[0]

    def forward_features(self,batch):
        """Return logits and a shared encoding; never retain batch state on the module."""
        if self.kind=='cnn':
            logits=self.cnn(batch['waveform']);return logits[:,1]-logits[:,0],None
        if self.kind=='tcn':
            v=self.encoder(batch['waveform']);return self.vibration_head(v).squeeze(-1),v
        mask=batch['vision_mask'];present=mask.any(1);z=self.vision_projection(batch['vision'])
        if self.kind=='vision':
            weights=mask.float()/mask.sum(1,keepdim=True).clamp_min(1)
            return self.vision_head((z*weights[:,:,None]).sum(1)).squeeze(-1),None
        tokens=self.encoder.tokens(batch['waveform']);v=tokens.mean(1)
        if self.fusion in ['attention','attention_gated']:
            safe_mask=mask.clone();safe_mask[~present,0]=True
            keys=z+self.offset_projection(batch['offsets'][:,:,None]/self.distance_scale)
            z,_=self.attention(tokens,keys,keys,key_padding_mask=~safe_mask,need_weights=False);visual=z.mean(1)
        else:
            if self.fusion in ['distance','gated']:
                weights=torch.exp(-.5*(batch['offsets']/self.distance_scale)**2)*mask
            elif self.fusion=='concat':
                distances=batch['offsets'].abs().masked_fill(~mask,float('inf'))
                indices=distances.argmin(1);weights=torch.zeros_like(distances).scatter_(1,indices[:,None],1.)*mask
            else:weights=mask.float()
            weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8);visual=(z*weights[:,:,None]).sum(1)
        visual=visual*present[:,None]
        if self.fusion in ['gated','attention_gated']:
            quality=(batch['quality']*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            gate=self.gate(torch.cat([v,visual,quality,present[:,None].float()],1));visual=visual*gate
        fused=self.head(torch.cat([v,visual,present[:,None].float()],1)).squeeze(-1)
        # Missing vision falls back to an explicitly supervised vibration head.
        return torch.where(present,fused,self.vibration_head(v).squeeze(-1)),v

    def auxiliary_loss(self,batch,labels,encoding=None):
        if self.kind!='fusion':return batch['waveform'].sum()*0
        v=self.encoder(batch['waveform']) if encoding is None else encoding
        return .25*nn.functional.binary_cross_entropy_with_logits(self.vibration_head(v).squeeze(-1),labels)

class TimeFrequencySSL(nn.Module):
    """VICReg-style time/frequency views; not a replica of a published method."""
    def __init__(self,cfg):
        super().__init__();m=cfg['model'];d=m['embedding_dim']
        self.encoder=TemporalEncoder(len(m['channels']),m['width'],m['depth'],m['kernel_size'],d,m['dropout'])
        self.frequency=nn.Sequential(nn.Linear(len(m['channels'])*128,d),nn.GELU(),nn.Linear(d,d))
        self.projector=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,d))
    def forward(self,waveform):
        time=self.projector(self.encoder(waveform))
        # Fixed-size log-amplitude representation, retain <= Nyquist/2 (500Hz at 2kHz).
        frequency=torch.log1p(torch.fft.rfft(waveform.float(),dim=-1).abs())
        frequency=frequency[...,:frequency.shape[-1]//2]
        frequency=nn.functional.adaptive_avg_pool1d(frequency,128).flatten(1)
        return time,self.frequency(frequency)

def ssl_loss(a,b,variance_weight=1.,covariance_weight=.05):
    if len(a)<2:raise ValueError('Self-supervision requires at least two examples per batch')
    invariance=nn.functional.mse_loss(a,b)
    variance=(torch.relu(1-torch.sqrt(a.var(0)+1e-4)).mean()+torch.relu(1-torch.sqrt(b.var(0)+1e-4)).mean())/2
    covariance=0.
    for z in [a,b]:
        z=z-z.mean(0);cov=(z.T@z)/(len(z)-1);off=cov-torch.diag(torch.diag(cov));covariance=covariance+off.square().sum()/z.shape[1]
    return invariance+variance_weight*variance+covariance_weight*covariance

def make_model(cfg):return ResearchModel(cfg)
