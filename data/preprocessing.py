import numpy as np
import pandas as pd
import polars as pl
import torch
import os
import hashlib
import pickle
from data.schemas import FUT_SUFFIX
from einops import rearrange
from sentence_transformers import SentenceTransformer
from typing import List


class PreprocessingMixin:
    @staticmethod
    def _get_dataset_identifier(dataset_name, dataset_split=None):
        """
        生成数据集标识符，用于缓存文件命名
        
        Args:
            dataset_name: 数据集名称 (amazon, ml1m, ml32m)
            dataset_split: 数据集分割名称 (beauty, sports, toys等，仅用于Amazon数据集)
            
        Returns:
            str: 数据集标识符
        """
        if dataset_name == "amazon" and dataset_split is not None:
            return f"amazon_{dataset_split}"
        else:
            return dataset_name
    @staticmethod
    def _process_genres(genres, one_hot=True):
        if one_hot:
            return genres

        max_genres = genres.sum(axis=1).max()
        idx_list = []
        for i in range(genres.shape[0]):
            idxs = np.where(genres[i, :] == 1)[0] + 1
            missing = max_genres - len(idxs)
            if missing > 0:
                idxs = np.array(list(idxs) + missing * [0])
            idx_list.append(idxs)
        out = np.stack(idx_list)
        return out

    @staticmethod
    def _remove_low_occurrence(source_df, target_df, index_col):
        if isinstance(index_col, str):
            index_col = [index_col]
        out = target_df.copy()
        for col in index_col:
            count = source_df.groupby(col).agg(ratingCnt=("rating", "count"))
            high_occ = count[count["ratingCnt"] >= 5]
            out = out.merge(high_occ, on=col).drop(columns=["ratingCnt"])
        return out

    @staticmethod
    def _encode_text_feature(text_feat, model=None, cache_dir="embeddings_cache", dataset_name=None, dataset_split=None):
        """
        对文本特征进行编码，如果embedding已存在则直接加载，否则生成并保存
        
        Args:
            text_feat: 文本特征列表或pandas Series
            model: SentenceTransformer模型，如果为None则使用默认模型
            cache_dir: 缓存目录路径
            dataset_name: 数据集名称，用于标识缓存文件。如果为None，则使用哈希值
            dataset_split: 数据集分割名称（如beauty、sports、toys等），用于Amazon数据集
            
        Returns:
            torch.Tensor: 编码后的embedding
        """
        if model is None:
            model = SentenceTransformer('sentence-transformers/sentence-t5-xxl')
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 将文本特征转换为列表
        if hasattr(text_feat, 'tolist'):
            text_list = text_feat.tolist()
        else:
            text_list = list(text_feat)
        
        # 根据是否提供数据集名称来决定缓存文件名
        if dataset_name is not None:
            # 使用辅助函数生成数据集标识符
            dataset_id = PreprocessingMixin._get_dataset_identifier(dataset_name, dataset_split)
            cache_filename = f"embeddings_{dataset_id}.pkl"
            cache_file = os.path.join(cache_dir, cache_filename)
        else:
            # 如果没有提供数据集名称，则使用哈希值作为备选方案
            text_content = "|".join(text_list)
            text_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
            cache_file = os.path.join(cache_dir, f"embeddings_{text_hash}.pkl")
        
        # 检查缓存是否存在
        if os.path.exists(cache_file):
            print(f"从缓存加载embedding: {cache_file}")
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
            return embeddings
        
        # 生成新的embedding
        print("生成新的embedding...")
        embeddings = model.encode(
            batch_size=1, 
            sentences=text_list, 
            show_progress_bar=True, 
            convert_to_tensor=True
        ).cpu()
        
        # 保存到缓存
        print(f"保存embedding到缓存: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        return embeddings
    
    @staticmethod
    def _rolling_window(group, features, window_size=200, stride=1):
        assert group["userId"].nunique() == 1, "Found data for too many users"
        
        if len(group) < window_size:
            window_size = len(group)
            stride = 1
        n_windows = (len(group)+1-window_size)//stride
        feats = group[features].to_numpy().T
        windows = np.lib.stride_tricks.as_strided(
            feats,
            shape=(len(features), n_windows, window_size),
            strides=(feats.strides[0], 8*stride, 8*1)
        )
        feat_seqs = np.split(windows, len(features), axis=0)
        rolling_df = pd.DataFrame({
            name: pd.Series(
                np.split(feat_seqs[i].squeeze(0), n_windows, 0)
            ).map(torch.tensor) for i, name in enumerate(features)
        })
        return rolling_df
    
    @staticmethod
    def _ordered_train_test_split(df, on, train_split=0.8):
        threshold = df.select(pl.quantile(on, train_split)).item()
        return df.with_columns(is_train=pl.col(on) <= threshold)
    
    @staticmethod
    def _df_to_tensor_dict(df, features):
        out = {
            feat: torch.from_numpy(
                rearrange(
                    df.select(feat).to_numpy().squeeze().tolist(), "b d -> b d"
                )
            ) if df.select(pl.col(feat).list.len().max() == pl.col(feat).list.len().min()).item()
            else df.get_column("itemId").to_list()
            for feat in features
        }
        fut_out = {
            feat + FUT_SUFFIX: torch.from_numpy(
                df.select(feat + FUT_SUFFIX).to_numpy()
            ) for feat in features
        }
        out.update(fut_out)
        out["userId"] = torch.from_numpy(df.select("userId").to_numpy())
        return out


    @staticmethod
    def _generate_user_history(
        ratings_df,
        features: List[str] = ["movieId", "rating"],
        window_size: int = 200,
        stride: int = 1,
        train_split: float = 0.8,
    ) -> torch.Tensor:
        
        if isinstance(ratings_df, pd.DataFrame):
            ratings_df = pl.from_pandas(ratings_df)

        grouped_by_user = (ratings_df
            .sort("userId", "timestamp")
            .group_by_dynamic(
                index_column=pl.int_range(pl.len()),
                every=f"{stride}i",
                period=f"{window_size}i",
                by="userId")
            .agg(
                *(pl.col(feat) for feat in features),
                seq_len=pl.col(features[0]).len(),
                max_timestamp=pl.max("timestamp")
            )
        )
        
        max_seq_len = grouped_by_user.select(pl.col("seq_len").max()).item()
        split_grouped_by_user = PreprocessingMixin._ordered_train_test_split(grouped_by_user, "max_timestamp", 0.8)
        padded_history = (split_grouped_by_user
            .with_columns(pad_len=max_seq_len-pl.col("seq_len"))
            .filter(pl.col("is_train").or_(pl.col("seq_len") > 1))
            .select(
                pl.col("userId"),
                pl.col("max_timestamp"),
                pl.col("is_train"),
                *(pl.when(pl.col("is_train"))
                    .then(
                        pl.col(feat).list.concat(
                            pl.lit(-1, dtype=pl.Int64).repeat_by(pl.col("pad_len"))
                        ).list.to_array(max_seq_len)
                    ).otherwise(
                        pl.col(feat).list.slice(0, pl.col("seq_len")-1).list.concat(
                            pl.lit(-1, dtype=pl.Int64).repeat_by(pl.col("pad_len")+1)
                        ).list.to_array(max_seq_len)
                    )
                    for feat in features
                ),
                *(pl.when(pl.col("is_train"))
                    .then(
                        pl.lit(-1, dtype=pl.Int64)
                    )
                    .otherwise(
                        pl.col(feat).list.get(-1)
                    ).alias(feat + FUT_SUFFIX)
                    for feat in features
                )
            )
        )
        
        out = {}
        out["train"] = PreprocessingMixin._df_to_tensor_dict(
            padded_history.filter(pl.col("is_train")),
            features
        )
        out["eval"] = PreprocessingMixin._df_to_tensor_dict(
            padded_history.filter(pl.col("is_train").not_()),
            features
        )
        
        return out

