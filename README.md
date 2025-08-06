
### CFN based on mlp
1. Make sure you have `torch` and `transformers`

2. Install cfn

```bash
cd your_path/VLA-tts
pip install -e .
```

3. Useage

```python
    from cfn.cfn_net import CFNWrapper
    import torch

    ### load model
    model = CFNWrapper(
        state_dim=state_dim,
        action_dim=cfn_action_steps * action_dim,
        language_model_name="bert-base-uncased",
        embed_dim=128,
        cfn_output_dim=20,
    ).to(device)

    weight_path = 'your_path/model_epoch1.pt'
    print(f"🔍 加载模型权重: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        ### Calculate norm
        state = batch['observation.state'].to(device)
        action = action[:, :cfn_action_steps, :].to(device)
        task = batch['task']
        batch_size = action.shape[0]
        action_flat = action.reshape(batch_size, -1)

        cfn_output = model(state, action_flat, task)

        ### 找最小norm对应的action
        
        norm = cfn_output.norm(dim=1)
        # 1. 找到最小值
        min_val = torch.min(norm)
        # 2. 找出所有等于最小值的索引
        indices = torch.nonzero(norm == min_val).squeeze()
        # 3. 从中随机选择一个索引
        if indices.ndim == 0:
            # 只有一个最小值的情况
            selected_index = indices.item()
        else:
            selected_index = indices[torch.randint(0, len(indices), (1,))].item()    

        outputs_action = batch['action'][[selected_index], ...]

```

