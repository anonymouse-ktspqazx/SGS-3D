import torch
import numpy as np
import os
import cv2
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import sys
import pickle

#### Foundation 2D
import clip
from util2d.openai_clip import CLIP_OpenAI
from util2d.segment_anything_hq import SAM_HQ

#### Grounding DINO
from detectron2.structures import BitMasks
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

#### Util
import supervision as sv
from util2d.util import show_mask, masks_to_rle
from util2d.mask_filtering_anonymous import mask_filter_anonymous

#### Dataset and mapping
from dataset import build_dataset
from dataset.scannet_loader import ScanNetReader, scaling_mapping
from src.fusion_util import NMS_cuda
from src.mapper import PointCloudToImageMapper


class GroundingDINO_SAM_Anonymous:
    """
    Anonymous implementation of Grounding DINO + SAM pipeline
    This is a simplified version for anonymous review
    """

    def __init__(self, cfg):
        # Load Foundation Models
        sam2d = SAM_HQ(cfg)
        clip2d = CLIP_OpenAI(cfg)
        self.sam_predictor = sam2d.sam_predictor
        self.clip_adapter, self.clip_preprocess = clip2d.clip_adapter, clip2d.clip_preprocess

        # Load Grounding DINO model
        self.grounding_dino_model = self.init_grounding_dino_model(cfg)

    def extract_masks_and_features(self, scene_id, class_names, cfg, gen_feat=True):
        """
        Anonymous implementation of 2D mask extraction
        
        Args:
            scene_id: Scene identifier
            class_names: List of class names for detection
            cfg: Configuration object
            gen_feat: Whether to generate features
            
        Returns:
            grounded_data_dict: Dictionary containing masks and metadata
            grounded_features: Point cloud features
        """
        scene_dir = os.path.join(cfg.data.datapath, scene_id)
        loader = build_dataset(root_path=scene_dir, cfg=cfg)

        # Initialize point cloud mapper
        img_dim = cfg.data.img_dim
        pointcloud_mapper = PointCloudToImageMapper(
            image_dim=img_dim, 
            intrinsics=loader.global_intrinsic, 
            cut_bound=cfg.data.cut_num_pixel_boundary
        )

        points = loader.read_pointcloud()
        points = torch.from_numpy(points).cuda()
        n_points = points.shape[0]

        grounded_data_dict = {}

        # Initialize feature accumulator
        if gen_feat:
            grounded_features = torch.zeros((n_points, cfg.foundation_model.clip_dim)).cuda()
        else:
            grounded_features = None

        # Create output directories for visualization
        vismask_dir = f"{cfg.exp.save_dir}/anonymous_2d_masks/{scene_id}/masks"
        os.makedirs(vismask_dir, exist_ok=True)
        
        frame_info_dict = {}
        skipnomask_frames = set()

        # Stage 1: Generate initial masks
        print("Stage 1: Generating initial masks...")
        for i in trange(0, len(loader), cfg.data.img_interval):
            frame = loader[i]
            frame_id = frame["frame_id"]
            image_path = frame["image_path"]

            image_pil = Image.open(image_path).convert("RGB")
            
            # Get detection boxes from Grounding DINO
            boxes_filt, confs_filt, labels_filt = self.detect_objects(
                image_path, class_names, cfg
            )
            
            if len(boxes_filt) == 0:
                skipnomask_frames.add(frame_id)
                continue

            # Generate masks using SAM
            masks = self.generate_masks_with_sam(image_path, boxes_filt, cfg)
            
            if masks is None or len(masks) == 0:
                skipnomask_frames.add(frame_id)
                continue

            # Process and filter masks
            frame_masks = self.process_masks(masks, boxes_filt, confs_filt, labels_filt)
            
            if not frame_masks:
                skipnomask_frames.add(frame_id)
                continue

            frame_info_dict[frame_id] = {'masks': frame_masks}
            
            # Save visualization
            self.save_mask_visualization(masks, vismask_dir, frame_id)

        # Stage 2: Filter masks (simplified version)
        print("Stage 2: Filtering masks...")
        frame_info_filt_dict = mask_filter_anonymous(cfg, scene_id, frame_info_dict, skipnomask_frames)

        # Stage 3: Extract features and save final results
        print("Stage 3: Extracting features...")
        for i in trange(0, len(loader), cfg.data.img_interval):
            frame = loader[i]
            frame_id = frame["frame_id"]
            image_path = frame["image_path"]
            
            if frame_id in skipnomask_frames or frame_id not in frame_info_filt_dict:
                continue

            # Load filtered masks and extract features
            filtered_masks_info = frame_info_filt_dict[frame_id]['masks']
            if not filtered_masks_info:
                continue

            # Reconstruct masks and metadata
            boxes_list, confs_list, labels_list, masks_list = self.reconstruct_masks(
                filtered_masks_info, vismask_dir, frame_id
            )
            
            if len(masks_list) == 0:
                continue

            # Extract CLIP features
            image_features = self.extract_clip_features(image_path, boxes_list, masks_list, cfg)

            # Save frame data
            grounded_data_dict[frame_id] = {
                "boxes": torch.tensor(boxes_list).cpu(),
                "masks": masks_to_rle(torch.stack(masks_list)),
                "class": labels_list,
                "img_feat": image_features.cpu(),
                "conf": torch.tensor(confs_list).cpu(),
            }

            # Accumulate features to point cloud
            if gen_feat:
                self.accumulate_features_to_pointcloud(
                    frame, loader, pointcloud_mapper, points, 
                    torch.stack(masks_list), image_features, 
                    grounded_features, cfg
                )

        return grounded_data_dict, grounded_features

    def detect_objects(self, image_path, class_names, cfg):
        """Simplified object detection using Grounding DINO"""
        image_pil = Image.open(image_path).convert("RGB")
        image_pil, image_infer = self.load_image(image_pil)

        boxes_filt = []
        confs_filt = []
        labels_filt = []

        # Split class names into chunks to avoid memory issues
        segment_size = 10
        segments = [class_names[i:i + segment_size] for i in range(0, len(class_names), segment_size)]

        for cls_chunk in segments:
            boxes, confidences = self.get_grounding_output(
                image_infer,
                ".".join(cls_chunk),
                cfg.foundation_model.box_threshold,
                cfg.foundation_model.text_threshold,
                device=cfg.foundation_model.device,
            )

            if len(boxes) > 0:
                boxes_filt.append(boxes)
                confs_filt.append(confidences)
                labels_filt.extend(cls_chunk)

        if len(boxes_filt) == 0:
            return [], [], []

        boxes_filt = torch.cat(boxes_filt)
        confs_filt = torch.cat(confs_filt)

        # Convert from normalized coordinates to pixel coordinates
        size = image_pil.size
        H, W = size[1], size[0]
        boxes_filt = boxes_filt * torch.Tensor([W, H, W, H])[None, ...].cuda()

        # Convert from XYWH to XYXY format
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
        boxes_filt[:, 2:] += boxes_filt[:, :2]

        # Clip boxes to image boundaries
        boxes_filt[:, 0] = boxes_filt[:, 0].clip(0)
        boxes_filt[:, 1] = boxes_filt[:, 1].clip(0)
        boxes_filt[:, 2] = boxes_filt[:, 2].clip(min=0, max=W)
        boxes_filt[:, 3] = boxes_filt[:, 3].clip(min=0, max=H)

        # Filter out invalid boxes
        valid_boxes = ((boxes_filt[:, 3] - boxes_filt[:, 1]) > 1) & \
                     ((boxes_filt[:, 2] - boxes_filt[:, 0]) > 1) & \
                     ((boxes_filt[:, 3] - boxes_filt[:, 1]) * (boxes_filt[:, 2] - boxes_filt[:, 0]) / (W * H) < 0.85)

        if valid_boxes.sum() == 0:
            return [], [], []

        boxes_filt = boxes_filt[valid_boxes]
        confs_filt = confs_filt[valid_boxes]
        labels_filt = [labels_filt[i] for i in range(len(labels_filt)) if valid_boxes[i]]

        return boxes_filt, confs_filt, labels_filt

    def generate_masks_with_sam(self, image_path, boxes, cfg):
        """Generate masks using SAM with box prompts"""
        image_sam = cv2.imread(image_path)
        image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
        
        self.sam_predictor.set_image(image_sam)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes, image_sam.shape[:2]
        )
        
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(cfg.foundation_model.device),
            multimask_output=False,
        )
        
        return masks.squeeze(1) if masks is not None else None

    def process_masks(self, masks, boxes, confs, labels):
        """Process and rank masks by confidence"""
        # Simplified mask processing
        frame_masks = {}
        mask_id_counter = 1
        
        # Sort by confidence
        _, ranks = torch.sort(confs.flatten(), descending=True)
        
        for index in ranks:
            if torch.sum(masks[index]) < 400:  # Filter small masks
                continue
                
            frame_masks[mask_id_counter] = {
                'box': boxes[index].cpu().numpy(),
                'confidence': confs[index].item(),
                'label': labels[index]
            }
            mask_id_counter += 1
            
        return frame_masks

    def extract_clip_features(self, image_path, boxes_list, masks_list, cfg):
        """Extract CLIP features for masked regions"""
        image_sam = cv2.imread(image_path)
        image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
        
        regions = []
        for box, mask in zip(boxes_list, masks_list):
            l, t, r, b = map(int, box)
            
            # Extract masked region
            tmp = torch.tensor(image_sam)[t:b, l:r, :].cuda()
            
            # Apply mask (simplified version)
            mask_region = mask[t:b, l:r]
            background_pixels = ~mask_region
            
            # Blur background
            tmp[background_pixels, :] = tmp[background_pixels, :] * 0.5
            
            regions.append(self.clip_preprocess(Image.fromarray(tmp.cpu().numpy())))

        # Extract CLIP features
        imgs = torch.stack(regions).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_adapter.encode_image(imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features

    def accumulate_features_to_pointcloud(self, frame, loader, pointcloud_mapper, 
                                        points, masks, image_features, 
                                        grounded_features, cfg):
        """Accumulate CLIP features to point cloud (simplified)"""
        # This is a simplified version of the feature accumulation
        # Full implementation details are omitted for anonymous review
        
        pose = loader.read_pose(frame["pose_path"])
        depth = loader.read_depth(frame["depth_path"])
        
        # Compute point-to-image mapping
        mapping = torch.ones([len(points), 4], dtype=int, device="cuda")
        mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(
            pose, points, depth, intrinsic=frame.get("translated_intrinsics")
        )
        
        idx = torch.where(mapping[:, 3] == 1)[0]
        
        if len(idx) < 100:
            return
            
        # Accumulate features (simplified)
        pred_masks = BitMasks(masks)
        final_feat = torch.einsum("qc,qhw->chw", image_features.float(), pred_masks.tensor.float())
        grounded_features[idx] += final_feat[:, mapping[idx, 1], mapping[idx, 2]].permute(1, 0)

    def get_grounding_output(self, image, caption, box_threshold, text_threshold, with_logits=True, device="cuda"):
        """
        Grounding DINO box generator
        Returning boxes and logits scores for each chunk in the caption with box & text threshoding
        """
        # Caption formatting
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        self.grounding_dino_model = self.grounding_dino_model.to(device)
        image = image.to(device)

        # Grounding DINO box generator
        with torch.no_grad():
            outputs = self.grounding_dino_model(image[None], captions=[caption])
            logits = outputs["pred_logits"].sigmoid()[0]  # (nqueries, 256)
        boxes = outputs["pred_boxes"][0]  # (nqueries, 4)

        # Filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        return boxes_filt, logits_filt.max(dim=1)[0]

    def init_grounding_dino_model(self, cfg):
        """Initialize Grounding DINO model"""
        grounding_dino_model = self.load_model(
            cfg.foundation_model.grounded_config_file,
            cfg.foundation_model.grounded_checkpoint,
            device="cuda"
        )
        print('------- Loaded Grounding DINO Model -------')
        return grounding_dino_model

    def load_model(self, model_config_path, model_checkpoint_path, device):
        """Grounding DINO loader"""
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        model.cuda()
        return model

    def load_image(self, image_pil):
        """Grounding DINO preprocess"""
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def save_mask_visualization(self, masks, output_dir, frame_id):
        """Save mask visualization"""
        if masks is None:
            return
            
        # Create visualization (simplified)
        mask_image = np.zeros(masks[0].shape, dtype=np.uint8)
        for i, mask in enumerate(masks):
            mask_image[mask.cpu().numpy()] = i + 1
            
        cv2.imwrite(os.path.join(output_dir, f"{frame_id}.png"), mask_image)

    def reconstruct_masks(self, filtered_masks_info, vismask_dir, frame_id):
        """Reconstruct masks from filtered information"""
        # Load saved mask image
        mask_path = os.path.join(vismask_dir, f"{frame_id}.png")
        if not os.path.exists(mask_path):
            return [], [], [], []
            
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        boxes_list = []
        confs_list = []
        labels_list = []
        masks_list = []
        
        for m_id in sorted(filtered_masks_info.keys()):
            boxes_list.append(filtered_masks_info[m_id]['box'])
            confs_list.append(filtered_masks_info[m_id]['confidence'])
            labels_list.append(filtered_masks_info[m_id]['label'])
            
            mask = torch.from_numpy(mask_image == m_id).cuda()
            masks_list.append(mask)
            
        return boxes_list, confs_list, labels_list, masks_list
