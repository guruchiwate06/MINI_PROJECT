"""Frozen reference-route projection, not a claim of surveyed GNSS accuracy."""
import numpy as np
from scipy.spatial import cKDTree
from .common import finite
from .errors import DataIntegrityError

def local_xy(latitude,longitude,origin):
    return np.column_stack([(np.asarray(longitude)-origin[1])*111320*np.cos(np.deg2rad(origin[0])),
                            (np.asarray(latitude)-origin[0])*111320])

class TrackReference:
    def __init__(self,latitude,longitude,origin=None):
        self.origin=list(origin if origin is not None else [float(latitude[0]),float(longitude[0])])
        line=local_xy(latitude,longitude,self.origin); finite(line,'reference route')
        line=line[np.r_[True,np.linalg.norm(np.diff(line,axis=0),axis=1)>.01]]
        if len(line)<3: raise DataIntegrityError('Reference route has fewer than three distinct vertices')
        self.line=line; self.vectors=np.diff(line,axis=0); self.lengths=np.linalg.norm(self.vectors,axis=1)
        self.chain=np.r_[0,np.cumsum(self.lengths)]
        self.tree=cKDTree((line[:-1]+line[1:])/2)

    def project(self,latitude,longitude):
        q=local_xy(latitude,longitude,self.origin); finite(q,'GPS coordinates')
        _,indices=self.tree.query(q,k=min(3,len(self.vectors)))
        if indices.ndim==1: indices=indices[:,None]
        starts=self.line[indices]; v=self.vectors[indices]
        u=np.clip(((q[:,None]-starts)*v).sum(-1)/(v*v).sum(-1),0,1)
        distance=np.linalg.norm(starts+u[:,:,None]*v-q[:,None],axis=-1)
        best=distance.argmin(1); rows=np.arange(len(q)); ii=indices[rows,best]
        return self.chain[ii]+u[rows,best]*self.lengths[ii],distance[rows,best]

def nearest_indices(sorted_values,queries):
    if len(sorted_values)==0: raise DataIntegrityError('Cannot match to an empty sequence')
    j=np.searchsorted(sorted_values,queries).clip(0,len(sorted_values)-1); k=(j-1).clip(0,len(sorted_values)-1)
    return np.where(abs(sorted_values[k]-queries)<abs(sorted_values[j]-queries),k,j)
