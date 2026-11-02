"""Half-open source-interval checks, independent of split names and labels."""
import numpy as np

def overlap_mask(candidate, reference):
    """Mark candidate windows sharing >=1 raw sample in the same run."""
    mask=np.zeros(len(candidate),dtype=bool)
    for run in candidate.run_id.unique():
        positions=np.flatnonzero(candidate.run_id.to_numpy()==run)
        other=reference[reference.run_id==run].sort_values('start_idx')
        if other.empty: continue
        starts=other.start_idx.to_numpy()
        cumulative_ends=np.maximum.accumulate(other.end_idx.to_numpy())
        c=candidate.iloc[positions]
        j=np.searchsorted(starts,c.end_idx.to_numpy(),side='left')-1
        valid=j>=0
        mask[positions[valid]]=cumulative_ends[j[valid]]>c.start_idx.to_numpy()[valid]
    return mask

def purge_training_and_validation(train,validation,test):
    """Preserve test identities; purge validation, then training, against others.
    Does not guarantee spatial independence or remove multimodal footprints.
    """
    validation=validation.loc[~overlap_mask(validation,test)].copy()
    import pandas as pd
    train=train.loc[~overlap_mask(train,pd.concat([validation,test]))].copy()
    return train,validation,test.copy()
