from __future__ import annotations
import hashlib, importlib.metadata, json, logging, os, platform, random, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .errors import ArtifactError, DataIntegrityError

LOG = logging.getLogger("railvivid")

def utc_now(): return datetime.now(timezone.utc).isoformat()

def jsonable(value):
    if isinstance(value, dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [jsonable(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, np.generic): return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def canonical(value): return json.dumps(jsonable(value),sort_keys=True,separators=(",",":"),allow_nan=False)
def fingerprint(value): return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    try:
        with Path(path).open('rb') as f:
            for chunk in iter(lambda:f.read(2**20),b''): h.update(chunk)
    except OSError as e: raise ArtifactError(f"Cannot hash {path}: {e}") from e
    return h.hexdigest()

def atomic_write(path, writer, binary=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb' if binary else 'w',**({} if binary else {'encoding':'utf-8','newline':''})) as f:
            writer(f); f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    except BaseException:
        Path(name).unlink(missing_ok=True); raise

def write_json(path,value):
    atomic_write(path,lambda f:json.dump(jsonable(value),f,indent=2,allow_nan=False))

def read_json(path):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError) as e: raise ArtifactError(f"Cannot read JSON {path}: {e}") from e

def write_csv(path,frame): atomic_write(path,lambda f:frame.to_csv(f,index=False))
def save_array(path,array): atomic_write(path,lambda f:np.save(f,array,allow_pickle=False),binary=True)

def require_columns(frame,columns,source):
    missing=set(columns)-set(frame.columns)
    if missing: raise DataIntegrityError(f"{source}: missing columns {sorted(missing)}")

def finite(array,source):
    if not np.isfinite(array).all(): raise DataIntegrityError(f"{source}: contains NaN/Inf; refusing to silently impute")

def source_hash():
    source=Path(__file__).parent
    roots=[source,source.parent/'src'];files={}
    for root in roots:
        for p in sorted(root.rglob('*.py')):
            name=root.name+'/'+str(p.relative_to(root)).replace('\\','/')
            files[name]=hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
    return fingerprint(files)

def environment():
    import torch
    packages={}
    for p in ['torch','torchvision','transformers','numpy','pandas','scipy','scikit-learn','Pillow']:
        try: packages[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: packages[p]=None
    source=Path(__file__).parent
    try: git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source.parent,stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError): git=None
    return {'timestamp':utc_now(),'python':sys.version,'platform':platform.platform(),'packages':packages,
        'torch_cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'git_commit':git,
        'pipeline_source_hash':source_hash()}

def seed_everything(seed,deterministic=True):
    # Must be set before CUDA BLAS initialization for deterministic CUDA GEMMs.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark=False

def select_device(name='cuda',require_cuda=True):
    import torch
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise ConfigurationErrorForDevice("CUDA is unavailable. Install a CUDA-enabled PyTorch build on the training machine; no CPU training fallback is performed.")
    if require_cuda and not name.startswith('cuda'):
        raise ConfigurationErrorForDevice("This training configuration requires CUDA. Inference/preparation can use CPU.")
    return torch.device(name)

# Keep device failure in the public configuration-error hierarchy.
from .errors import ConfigurationError as ConfigurationErrorForDevice
