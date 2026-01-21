import os
# from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)

    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            pr(123)
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    # else:
    #     vision_tower = '/home/jovyan/workspace/code-th/honor-large-model/test_longtail_pretrain/models/sow_10w_his_allintent_dro_flash_35seq_len30pred_len1lr0.0001n_layers12dim192.pt'

    #     is_absolute_path_exists = os.path.exists(vision_tower)
    #     use_s2 = getattr(vision_tower_cfg, 's2', False)
    
    #     if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
    #         if use_s2:
    #             pr(123)
    #             return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
    #         else:
    #             return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
