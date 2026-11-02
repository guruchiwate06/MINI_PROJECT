from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageEnhance,ImageFilter
from scipy.ndimage import laplace
from numpy.lib.format import open_memmap
from .common import file_hash,fingerprint,finite,read_json,select_device,write_csv,write_json,utc_now
from .errors import ArtifactError,ConfigurationError,DataIntegrityError
from .prepare import load_prepared

LOG=logging.getLogger('railvivid')

def vision_identity(cfg,corruption=None):
    _,meta=load_prepared(cfg)
    return fingerprint({'data':meta['data_fingerprint'],'vision':cfg['vision'],'corruption':corruption or {}})

def vision_path(cfg,corruption=None):
    root,_=load_prepared(cfg)
    return root/cfg['vision']['embedding_dir']/vision_identity(cfg,corruption)[:20]

def extract_vision(cfg,device='cuda',allow_download=False,corruption=None):
    import torch
    try:from transformers import AutoImageProcessor,AutoModel
    except ImportError as e:raise ConfigurationError('Install the vision extra: pip install -e .[vision]') from e
    v=cfg['vision'];corruption=corruption or {};root,meta=load_prepared(cfg)
    if not v['revision'] and not Path(v['model_id']).is_dir():
        raise ConfigurationError('Pin vision.revision to the exact pretrained-model commit before feature extraction')
    target=vision_path(cfg,corruption)
    if target.exists():raise ArtifactError(f'Vision destination exists; never overwrite a feature cache: {target}')
    frames=pd.read_csv(root/'frames.csv',keep_default_na=False)
    frames=frames.loc[frames.rejection_reason.eq('')&~frames.run_id.isin(cfg['alignment']['excluded_image_runs'])].copy().reset_index(drop=True)
    if frames.empty:raise DataIntegrityError('No accepted camera frames to encode')
    dev=select_device(device,require_cuda=False)
    try:
        kwargs={'revision':v['revision'],'local_files_only':not allow_download,'trust_remote_code':False}
        processor=AutoImageProcessor.from_pretrained(v['model_id'],**kwargs)
        model=AutoModel.from_pretrained(v['model_id'],**kwargs).to(dev).eval()
    except (OSError,ValueError,RuntimeError) as e:
        raise ArtifactError('Cannot load the pinned vision encoder. On the training machine cache its authorized weights or explicitly use --allow-download. Check access, revision and installed Transformers version. '+str(e)) from e
    target.mkdir(parents=True,exist_ok=False)
    write_json(target/'INCOMPLETE.json',{'created':utc_now(),'model_id':v['model_id'],'revision':v['revision']})
    embeddings=patches=quality=None;hashes=[];patch_shape=None
    try:
        with torch.inference_mode():
            for start in range(0,len(frames),v['batch_size']):
                batch=frames.iloc[start:start+v['batch_size']];images=[];qualities=[]
                for row in batch.itertuples():
                    source=Path(cfg['data']['root'])/row.relative_path
                    try:
                        digest=file_hash(source)
                        with Image.open(source) as original:
                            original.load();image=original.convert('RGB')
                        w,h=image.size;draw=ImageDraw.Draw(image)
                        for box in v['mask_boxes']:draw.rectangle([round(box[0]*w),round(box[1]*h),round(box[2]*w),round(box[3]*h)],fill=(0,0,0))
                        box=v['roi'];image=image.crop((round(box[0]*w),round(box[1]*h),round(box[2]*w),round(box[3]*h)))
                        if corruption.get('blur_radius_px'):image=image.filter(ImageFilter.GaussianBlur(float(corruption['blur_radius_px'])))
                        if 'brightness_factor' in corruption:image=ImageEnhance.Brightness(image).enhance(float(corruption['brightness_factor']))
                        gray=np.asarray(image.convert('L').resize((224,224)),dtype=float)/255
                        qualities.append([float(gray.mean()),float(np.log1p(laplace(gray).var()))]);images.append(image);hashes.append(digest)
                    except (OSError,ValueError,ArtifactError) as e:raise DataIntegrityError(f'Frame decode failed: {source}. No samples were silently dropped. {e}') from e
                inputs=processor(images=images,return_tensors='pt').to(dev)
                output=model(**inputs);tokens=output.last_hidden_state
                if tokens.ndim!=3:raise ConfigurationError('This extractor supports token-based DINOv2/DINOv3 ViT encoders; use a dedicated adapter for convolutional encoders')
                registers=int(getattr(model.config,'num_register_tokens',0))
                patch=tokens[:,1+registers:];pooled=tokens[:,0]
                if pooled.shape[1]!=cfg['model']['vision_dim']:raise ConfigurationError(f'Encoder dimension {pooled.shape[1]} differs from model.vision_dim')
                if embeddings is None:
                    embeddings=open_memmap(target/'embeddings.npy',mode='w+',dtype='float32',shape=(len(frames),pooled.shape[1]))
                    quality=open_memmap(target/'quality.npy',mode='w+',dtype='float32',shape=(len(frames),2));patch_shape=list(patch.shape[1:])
                    if v['store_patch_tokens']:patches=open_memmap(target/'patches.npy',mode='w+',dtype='float16',shape=(len(frames),*patch_shape))
                stop=start+len(batch);a=pooled.float().cpu().numpy();finite(a,'vision encoder output');embeddings[start:stop]=a;quality[start:stop]=qualities
                if patches is not None:
                    pp=patch.float().cpu().numpy();finite(pp,'vision patch tokens');patches[start:stop]=pp
                LOG.info('Encoded frames %d/%d',stop,len(frames))
        embeddings.flush();quality.flush()
        if patches is not None:patches.flush()
        frames['image_sha256']=hashes
        # Detect content duplicates even when filenames differ. Abort rather than leaking across future folds.
        duplicates=int(frames.image_sha256.duplicated().sum())
        if duplicates:
            duplicate=frames.loc[frames.image_sha256.duplicated(keep=False)]
            write_csv(target/'content_duplicates.csv',duplicate)
            LOG.warning('%d duplicate image contents catalogued; training preflight rejects any cross-partition reuse',duplicates)
        write_csv(target/'frames.csv',frames);write_json(target/'processor.json',processor.to_dict())
        artifacts={p.name:file_hash(p) for p in target.iterdir() if p.is_file() and p.name!='INCOMPLETE.json'}
        result={'created':utc_now(),'cache_fingerprint':vision_identity(cfg,corruption),'data_fingerprint':meta['data_fingerprint'],
            'model_id':v['model_id'],'revision':v['revision'],'vision_config':v,'corruption':corruption,'frames':len(frames),
            'embedding_dim':int(embeddings.shape[1]),'patch_shape':patch_shape,'artifacts':artifacts,'duplicate_contents':duplicates,
            'scope':'stateless frozen pretrained encoding; reference-memory fitting remains training-only'}
        write_json(target/'READY.json',result);(target/'INCOMPLETE.json').unlink();return result
    except BaseException as e:
        write_json(target/'FAILURE.json',{'created':utc_now(),'error':str(e),'type':type(e).__name__})
        raise
