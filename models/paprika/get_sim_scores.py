from tqdm import tqdm
import pickle
from pathlib import Path
import numpy as np

from trainer import PaprikaConfig

from .helper import cos_sim, get_step_des_feats, get_all_video_ids


def gather_all_frame_S3D_embeddings(cfg: PaprikaConfig):
    videos = get_all_video_ids(cfg)

    frame_embeddings = []
    frame_lookup_table = []

    videos_missing_features = set()
    for v in tqdm(videos):
        try:
            video_s3d = np.load(
                Path(cfg.dataset.base_dir) / cfg.dataset.name / "features" / f"{v}.npy"
            )
            # video_s3d shape: (num_clips, num_subclips, 512)

            for c_idx in range(video_s3d.shape[0]):
                frame_embeddings.append(np.float64(np.mean(video_s3d[c_idx], axis=0)))
                frame_lookup_table.append((v, c_idx))

        except FileNotFoundError:
            videos_missing_features.add(v)

    if len(videos_missing_features) > 0:
        with open("videos_missing_features.pickle", "wb") as f:
            pickle.dump(videos_missing_features, f)
    assert len(videos_missing_features) == 0, (
        "There are videos missing features! "
        + "Please check saved videos_missing_features.pickle."
    )

    frame_embeddings = np.array(frame_embeddings)

    # segment video embeddings shape: (3741608, 512) for the subset
    # segment video embeddings shape: (51053844, 512) for the fullset
    return frame_embeddings, frame_lookup_table


def gather_all_narration_MPNet_embeddings(cfg: PaprikaConfig):
    videos = get_all_video_ids(cfg)

    narration_embeddings = []
    narration_lookup_table = []
    videos_missing_features = set()

    for v in tqdm(videos):
        try:
            text_mpnet = np.load(
                Path(cfg.dataset.base_dir)
                / cfg.dataset.name
                / "feats"
                / v
                / "text_mpnet.npy"
            )
            # text_mpnet shape: (num_clips, num_subclips, 768)
            for c_idx in range(text_mpnet.shape[0]):
                narration_embeddings.append(np.mean(text_mpnet[c_idx], axis=0))
                narration_lookup_table.append((v, c_idx))
        except FileNotFoundError:
            videos_missing_features.add(v)

    narration_embeddings = np.array(narration_embeddings)
    return narration_embeddings, narration_lookup_table


def find_step_similarities_for_segments_using_frame(
    cfg: PaprikaConfig,
    step_des_feats,
    segment_video_embeddings,
    segment_video_lookup_table,
):
    for segment_id in tqdm(range(len(segment_video_embeddings))):
        v, cidx = segment_video_lookup_table[segment_id]
        save_path = Path(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/sim_scores/{v}/segment_{cidx}.npy"
        )
        if not save_path.exists():
            # dot product as similarity score
            sim_scores = np.einsum(
                "ij,ij->i",
                step_des_feats,
                segment_video_embeddings[segment_id][np.newaxis, ...],
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, sim_scores)


def find_step_similarities_for_segments_using_narration(
    cfg: PaprikaConfig,
    step_des_feats,
    segment_narration_embeddings,
    segment_narration_lookup_table,
):
    for segment_id in tqdm(range(len(segment_narration_embeddings))):
        v, cidx = segment_narration_lookup_table[segment_id]
        save_path = Path(
            f"{cfg.dataset.base_dir}/{cfg.dataset.name}/sim_scores/{v}/segment_{cidx}.npy"
        )
        if not save_path.exists():
            cos_scores = cos_sim(
                step_des_feats,
                segment_narration_embeddings[segment_id],
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, cos_scores[:, 0].numpy())


def get_sim_scores(cfg: PaprikaConfig):
    step_des_feats = get_step_des_feats(cfg, language_model="S3D")
    segment_video_embeddings, segment_video_lookup_table = (
        gather_all_frame_S3D_embeddings(cfg)
    )
    find_step_similarities_for_segments_using_frame(
        cfg, step_des_feats, segment_video_embeddings, segment_video_lookup_table
    )


def get_DS_sim_scores(cfg: PaprikaConfig):
    step_des_feats = get_step_des_feats(cfg, language_model="MPNet")
    segment_narration_embeddings, segment_narration_lookup_table = (
        gather_all_narration_MPNet_embeddings(cfg)
    )
    find_step_similarities_for_segments_using_narration(
        cfg,
        step_des_feats,
        segment_narration_embeddings,
        segment_narration_lookup_table,
    )
