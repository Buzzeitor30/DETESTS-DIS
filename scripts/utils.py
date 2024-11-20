from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import re, emoji
import numpy as np
import torch, random


def get_kf_splits(pd_df, target_label, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    if target_label.startswith("implicit"):
        pd_df = pd_df[pd_df["stereotype"] == 1]
        values = pd_df[target_label].values.tolist()

    if isinstance(values[0], list):
        values = list(map(lambda x: str(x), values))
    le = LabelEncoder()
    ys = le.fit_transform(values)

    return skf.split(pd_df["text"], ys)


def preprocess_text(sample):
    def remove_users(text):
        return re.sub(r"@user", r"", text)

    def remove_url(text):
        return re.sub(r"url", r"", text)

    def remove_hashtags(text):
        return re.sub(r"#(\w+)", r"", text)

    def remove_emojis(text):
        return emoji.replace_emoji(text, replace=" ")

    text = sample["text"].lower()

    text = remove_users(text)
    return text


def confidence_interval_cross_validation(scores, z=1.96):
    std_dev = np.std(scores, ddof=1)
    # Calculate margin of error (ME)
    margin_of_error = z * std_dev / np.sqrt(len(scores))
    return margin_of_error


def set_torch_np_random_rseed(rseed=42):
    np.random.seed(rseed)
    random.seed(rseed)
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    torch.cuda.manual_seed_all(rseed)
