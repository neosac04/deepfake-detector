from backend.config import load_config


def test_load_config():
    cfg = load_config("backend/config/default.yaml")
    assert cfg.model.name in {"resnext_lstm", "efficientnet_gru"}
    assert cfg.model.num_classes == 2
    assert cfg.data.sequence_length > 0
