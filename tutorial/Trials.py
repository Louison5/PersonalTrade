# model2 = Stok(env.observation_space, env.action_space, int(env.action_space.shape[0]), model_config=MODEL_DEFAULTS, name="fuck")
# print(model2)
from ray.rllib.agents.pg import PGTrainer 
from ray.tune.registry import register_env
import ray
ModelCatalog.register_custom_model(model_name="cust_model", model_class=Stok)
trainer_cfg = dict(
    rollout_fragment_length = 200,
    # train_batch_size = 5, 
    explore = False,
    framework = "torch",
    model = {
    
    #尝试了以下三种模型：

    ##1 === Built-in options === 简单全连接
    ## FullyConnectedNetwork (tf and torch): rllib.models.tf|torch.fcnet.py
    ## These are used if no custom model is specified and the input space is 1D.
    ## Number of hidden layers to be used.
    #"fcnet_hiddens": [256, 256],
    ## Activation function descriptor.
    ## Supported values are: "tanh", "relu", "swish" (or "silu"),
    ## "linear" (or None).
    #"fcnet_activation": "tanh",


    #2 == LSTM ==
    # Whether to wrap the model with an LSTM.
    "use_lstm": True ,
    # Max seq len for training the LSTM, defaults to 20.
    "max_seq_len": 20,
    # Size of the LSTM cell.
    "lstm_cell_size": 256,
    # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
    "lstm_use_prev_action": False,
    # Whether to feed r_{t-1} to LSTM.
    "lstm_use_prev_reward": False,
    # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
    "_time_major": False,


    ##3 == Attention Nets (experimental: torch-version is untested) ==
    ## Whether to use a GTrXL ("Gru transformer XL"; attention net) as the
    ## wrapper Model around the default Model.
    #"use_attention": True,
    ## The number of transformer units within GTrXL.
    ## A transformer unit in GTrXL consists of a) MultiHeadAttention module and
    ## b) a position-wise MLP.
    #"attention_num_transformer_units": 1,
    ## The input and output size of each transformer unit.
    #"attention_dim": 64,
    ## The number of attention heads within the MultiHeadAttention units.
    #"attention_num_heads": 1,
    ## The dim of a single head (within the MultiHeadAttention units).
    #"attention_head_dim": 32,
    ## The memory sizes for inference and training.
    #"attention_memory_inference": 50,
    #"attention_memory_training": 50,
    ## The output dim of the position-wise MLP.
    #"attention_position_wise_mlp_dim": 32,
    ## The initial bias values for the 2 GRU gates within a transformer unit.
    #"attention_init_gru_gate_bias": 2.0,
    ## Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
    #"attention_use_n_prev_actions": 0,
    ## Whether to feed r_{t-n:t-1} to GTrXL.
    #"attention_use_n_prev_rewards": 0,


    }
    
)
ray.init(ignore_reinit_error=True)
trainer_cfg["env"] = "portfolio_real_management"
trainer_cfg["env_config"] = dict(dataset=dataset, task="train", device="cpu")
register_env("portfolio_real_management", lambda config: PortfolioManagementEnvironment(config))
pg_trainer = PGTrainer(trainer_cfg, env="portfolio_real_management")
print(pg_trainer.get_policy().model)