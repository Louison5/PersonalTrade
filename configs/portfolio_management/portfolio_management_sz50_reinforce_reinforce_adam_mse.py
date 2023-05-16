task_name = "portfolio_management"
dataset_name = "sz50"
net_name = "deeptrader"
agent_name = "pg"
optimizer_name = "adam"
loss_name = "mse"
work_dir = f"work_dir/{task_name}_{dataset_name}_{net_name}_{agent_name}_{optimizer_name}_{loss_name}"

_base_ = [
    f"../_base_/datasets/{task_name}/{dataset_name}.py",
    f"../_base_/environments/{task_name}/env.py",
    f"../_base_/trainers/{task_name}/reinforce_trainer.py",
    f"../_base_/losses/{loss_name}.py",
    f"../_base_/optimizers/{optimizer_name}.py",
    f"../_base_/nets/{net_name}.py",
]

environment = dict(type='PortfolioManagementDeepTraderEnvironment')
data = dict(
    type = "PortfolioManagementDataset",
    data_path = "data/portfolio_management/sz50",
    train_path = "data/portfolio_management/sz50/train.csv",
    valid_path = "data/portfolio_management/sz50/valid.csv",
    test_path = "data/portfolio_management/sz50/test.csv",
    test_dynamic_path='data/portfolio_management/sz50/test.csv',
    tech_indicator_list = [
        "zopen",
        "zhigh",
        "zlow",
        "zadjcp",
        "zclose",
        "zd_5",
        "zd_10",
        "zd_15",
        "zd_20",
        "zd_25",
        "zd_30"
    ],
    initial_amount = 100000,
    transaction_cost_pct = 0.001
)

trainer = dict(
    agent_name = agent_name,
    work_dir = work_dir,
    epochs=1000,
    configs = dict(
        rollout_fragment_length = 500,
        explore = True
    ),
)