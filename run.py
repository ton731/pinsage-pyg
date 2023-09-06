# %%
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# %% [markdown]
# reference: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html

# %% [markdown]
# ### Creating Heterogeneous Graph

# %%
data = HeteroData()

num_users = 10
num_items = 100
num_features_user = 5
num_features_item = 7
num_transactions = 100
num_features_transaction = 3

data['user'].x = torch.rand((num_users, num_features_user))
data['item'].x = torch.rand((num_items, num_features_item))

data['user', 'buys', 'item'].edge_index = torch.stack([
    torch.randint(high=num_users, size=(num_transactions,)),
    torch.randint(high=num_items, size=(num_transactions,))
])

data['user', 'buys', 'item'].edge_attr = torch.rand((num_transactions, num_features_transaction))

data = T.ToUndirected()(data)

print(data)

# %% [markdown]
# ### Creating Heterogeneous GNNs
# PyG provides three ways to create models on heterogenrous graph data:
# 1. `torch_geometric.nn.to_hetero()` to convert model
# 2. `conv.HeteroConv` to define individual functions for different types
# 3. Deploy existing heterogeneous GNN operators
# 
# For me, the second one is more clear and easy to understand, so I will use it in the following exampe.

# %%
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv

class HeteroGNN(torch.nn.Module):
    def __init__(self, user_dim, item_dim, transaction_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        
        self.lin_proj_user = Linear(user_dim, hidden_dim)
        self.lin_proj_item = Linear(item_dim, hidden_dim)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # ('user', 'buys', 'item'): SAGEConv(hidden_dim, hidden_dim),
                ('user', 'buys', 'item'): GATv2Conv(hidden_dim, hidden_dim, heads=2, edge_dim=transaction_dim),
                ('item', 'rev_buys', 'user'): GATv2Conv(hidden_dim, hidden_dim, heads=2, edge_dim=transaction_dim)
                # ('item', 'rev_buys', 'user'): GATv2Conv(hidden_dim, hidden_dim,)
            })
            self.convs.append(conv)
        
        self.trans_user = Linear(hidden_dim, output_dim)
        self.trans_item = Linear(hidden_dim, output_dim)
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # linear projections
        for node_type, x in x_dict.items():
            if node_type == 'user':
                x_dict[node_type] = self.lin_proj_user(x)
            elif node_type == 'item':
                x_dict[node_type] = self.lin_proj_item(x)
        
        # message passing convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, **{'edge_attr_dict': edge_attr_dict})
            # x_dict = conv(x_dict, edge_index_dict, 
            #               **{'edge_type1': edge_attr_dict[('user', 'buys', 'item')],
            #                'edge_type2': edge_attr_dict[('item', 'rev_buys', 'user')]})
            # x_dict = conv(x_dict, edge_index_dict, **edge_attr_dict)
            # x_dict = conv(x_dict, edge_index_dict)
            x_dict = {node_type: x.relu() for node_type, x in x_dict.items()}
        
        # final transformation
        for node_type, x in x_dict.items():
            if node_type == 'user':
                x_dict[node_type] = self.trans_user(x)
            elif node_type == 'item':
                x_dict[node_type] = self.trans_item(x)
        
        return x_dict


model_kwargs = {
    'user_dim': num_features_user,
    'item_dim': num_features_item,
    'transaction_dim': num_features_transaction,
    'hidden_dim': 32,
    'output_dim': 32,
    'num_layers': 2
}
model = HeteroGNN(**model_kwargs)
print(model)

# %%
output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

# %%
print(output['item'].shape)
print(output['user'].shape)

# %%



