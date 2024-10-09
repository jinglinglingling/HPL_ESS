from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DSECDataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, split, **kwargs):
        # assert kwargs.get('split') in [None, 'train']
        # if 'split' in kwargs:
        #     kwargs.pop('split')
        super(DSECDataset, self).__init__(
            img_suffix='.npy',
            seg_map_suffix='_labelTrainIds.png',
            split=split,
            **kwargs)

@DATASETS.register_module()
class DSEC_E2vidDataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self, split, **kwargs):
        # assert kwargs.get('split') in [None, 'train']
        # if 'split' in kwargs:
        #     kwargs.pop('split')
        super(DSEC_E2vidDataset, self).__init__(
            img_suffix='.npy',
            seg_map_suffix='.png',
            split=split,
            **kwargs)