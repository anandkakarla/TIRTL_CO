#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
from myDARNN import *


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# In[7]:


TS = pd.read_csv("bigmat_mn_60s.csv", parse_dates=['T'])
# ssallT = pd.read_csv("TIRTL_hist_bigclasmat.csv", parse_dates=['T'])


# In[35]:


plt.plot(TS['T'],TS['Tbig_hist'])


# In[8]:


TSn=TS.copy()
TSn=TSn.iloc[:,1:].rolling(10,center=True,min_periods=2).mean()
TSn['T']=TS['T']
TSn['S1A']=0.5*(TSn['CO_244_IE']+TSn['CO_456_IE'])
TSn['S2A']=0.5*(TSn['CO_234_TT']+TSn['CO_451_TT'])
TSn['Sdif']=TSn['S2A']-TSn['S1A']
TSn.loc[TSn['Sdif']==0,'Sdif']=np.nan
TSn.loc[TSn['Tbig_hist']==0,'Tbig_hist']=np.nan
TSn=TSn.loc[TSn['T'] < '2021-12-28 09:00:00',:]
ttmp=TSn.copy()
TSn.drop(list(TSn.filter(regex=("CO_*")).columns)+['T_hist','T_avgspd','AT', 'RH', 'WS', 'SR'],axis=1,inplace=True)
data=TSn.copy()
data['Day'] = data['T'].dt.day
data['Weekday'] = data['T'].dt.weekday  # Monday=0, Sunday=6
data['Hour'] = data['T'].dt.hour
data['Month'] = data['T'].dt.month
data.drop(columns=['S1A','S2A','T'],inplace=True)


batch_size = 128
timesteps = 16
n_timeseries = data.shape[1] - 1
train_brdr = int(data.shape[0]*0.7)
val_brdr = int(data.shape[0]*0.85)
# test_brdr= int(data.shape[0]*0.85)
targetC = "Sdif"


device = torch.device('cpu')
# device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


# In[40]:


X = np.zeros((len(data), timesteps, data.shape[1]-1))
y = np.zeros((len(data), timesteps, 1))


# In[41]:


for i, name in enumerate(list(data.columns[:-1])):
    for j in range(timesteps):
        X[:, j, i] = data[name].shift(timesteps - j - 1).fillna(method="bfill")


# In[42]:


for j in range(timesteps):
    y[:, j, 0] = data[targetC].shift(timesteps - j - 1).fillna(method="bfill")


# In[43]:


prediction_horizon = 15
target = data[targetC].shift(-prediction_horizon).fillna(method="ffill").values


# In[44]:


X = X[timesteps:]
y = y[timesteps:]
target = target[timesteps:]


# In[45]:


X_train = X[:train_brdr]
y_his_train = y[:train_brdr]
X_val = X[train_brdr:val_brdr]
y_his_val = y[train_brdr:val_brdr]
X_test = X[val_brdr:]
y_his_test = y[val_brdr:]
target_train = target[:train_brdr]
target_val = target[train_brdr:val_brdr]
target_test = target[val_brdr:]


# In[46]:


X_train_max = X_train.max(axis=0)
X_train_min = X_train.min(axis=0)
y_his_train_max = y_his_train.max(axis=0)
y_his_train_min = y_his_train.min(axis=0)
target_train_max = target_train.max(axis=0)
target_train_min = target_train.min(axis=0)


# In[47]:


X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
X_val = (X_val - X_train_min) / (X_train_max - X_train_min)
X_test = (X_test - X_train_min) / (X_train_max - X_train_min)

y_his_train = (y_his_train - y_his_train_min) / (y_his_train_max - y_his_train_min)
y_his_val = (y_his_val - y_his_train_min) / (y_his_train_max - y_his_train_min)
y_his_test = (y_his_test - y_his_train_min) / (y_his_train_max - y_his_train_min)

target_train = (target_train - target_train_min) / (target_train_max - target_train_min)
target_val = (target_val - target_train_min) / (target_train_max - target_train_min)
target_test = (target_test - target_train_min) / (target_train_max - target_train_min)


# ### DARNN

# In[48]:


X_train_t = torch.Tensor(X_train)
X_val_t = torch.Tensor(X_val)
X_test_t = torch.Tensor(X_test)
y_his_train_t = torch.Tensor(y_his_train)
y_his_val_t = torch.Tensor(y_his_val)
y_his_test_t = torch.Tensor(y_his_test)
target_train_t = torch.Tensor(target_train)
target_val_t = torch.Tensor(target_val)
target_test_t = torch.Tensor(target_test)


# In[49]:


model = DARNN(X_train.shape[2], 64, 64, X_train.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)


# In[50]:


epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.9)


# In[51]:


from torch.utils.data import TensorDataset, DataLoader
data_train_loader = DataLoader(TensorDataset(X_train_t, y_his_train_t, target_train_t), shuffle=True, batch_size=32)
data_val_loader = DataLoader(TensorDataset(X_val_t, y_his_val_t, target_val_t), shuffle=False, batch_size=32)
data_test_loader = DataLoader(TensorDataset(X_test_t, y_his_test_t, target_test_t), shuffle=False, batch_size=32)


# In[52]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[53]:


epochs = 5
loss = nn.MSELoss()
patience = 8
min_val_loss = 9999
counter = 0
for i in range(epochs):
    mse_train = 0
    for batch_x, batch_y_h, batch_y in data_train_loader :
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_h = batch_y_h.cpu()
        opt.zero_grad()
        y_pred = model(batch_x, batch_y_h)
        y_pred = y_pred.squeeze(1)
        l = loss(y_pred, batch_y)
        l.backward()
        mse_train += l.item()*batch_x.shape[0]
        opt.step()
    epoch_scheduler.step()
    with torch.no_grad():
        mse_val = 0
        preds = []
        true = []
        for batch_x, batch_y_h, batch_y in data_val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y_h = batch_y_h.cpu()
            output = model(batch_x, batch_y_h)
            output = output.squeeze(1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            mse_val += loss(output, batch_y).item()*batch_x.shape[0]
    preds = np.concatenate(preds)
    true = np.concatenate(true)
    
    if min_val_loss > mse_val**0.5:
        min_val_loss = mse_val**0.5
        print("Saving...")
        torch.save(model.state_dict(), "iith.pt")
        counter = 0
    else: 
        counter += 1
    
    if counter == patience:
        break
    print("Iter: ", i, "train: ", (mse_train/len(X_train_t))**0.5, "val: ", (mse_val/len(X_val_t))**0.5)
#     if(i % 10 == 0):
#         preds = preds*(target_train_max - target_train_min) + target_train_min
#         true = true*(target_train_max - target_train_min) + target_train_min
#         mse = mean_squared_error(true, preds)
#         mae = mean_absolute_error(true, preds)
#         print("mse: ", mse, "mae: ", mae)
#         plt.figure(figsize=(20, 10))
#         plt.plot(preds)
#         plt.plot(true)
#         plt.show()


# In[55]:


model.load_state_dict(torch.load("iith.pt"))


# In[60]:


with torch.no_grad():
    mse_val = 0
    preds = []
    true = []
    for batch_x, batch_y_h, batch_y in data_test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_h = batch_y_h.to(device)
        output = model(batch_x, batch_y_h)
        preds.append(output.detach().cpu().numpy())
        true.append(batch_y.detach().cpu().numpy())
        mse_val += loss(output, batch_y).item()*batch_x.shape[0]
preds = np.concatenate(preds)
true = np.concatenate(true)


# In[61]:


preds = preds*(target_train_max - target_train_min) + target_train_min
true = true*(target_train_max - target_train_min) + target_train_min
fre=[(x,y[0]) for x,y in zip(true,preds) if not np.isnan(y)]
mt=np.vstack(np.array(fre))
mse = mean_squared_error(mt[0], mt[1])
mae = mean_absolute_error(mt[0], mt[1])
mape = mean_absolute_percentage_error(mt[0], mt[1])
r2_score = r2_score(mt[0], mt[1])
mse, mae, mape,r2_score

# In[29]:




# In[58]:


# from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
# import joblib
# (preds,true)=joblib.load('vfr_delS_10T')
# fre=[(x,y[0]) for x,y in zip(true,preds) if not np.isnan(y)]
# mt=np.vstack(np.array(fre))
# mse = mean_squared_error(mt[0], mt[1])
# mae = mean_absolute_error(mt[0], mt[1])
# mape = mean_absolute_percentage_error(mt[0], mt[1])
# r2_score = r2_score(mt[0], mt[1])
# mse, mae, mape,r2_score


# # In[36]:


# from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
# import joblib
# (preds,true)=joblib.load('vfr_s2_15T')
# fre=[(x,y[0]) for x,y in zip(true,preds) if not np.isnan(y)]
# mt=np.vstack(np.array(fre))
# mse = mean_squared_error(mt[0], mt[1])
# mae = mean_absolute_error(mt[0], mt[1])
# mape = mean_absolute_percentage_error(mt[0], mt[1])
# r2_score = r2_score(mt[0], mt[1])
# mse, mae, mape,r2_score


# # In[10]:


# from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
# import joblib
# (preds,true)=joblib.load('vfr_s1_s2_15T')
# fre=[(x,y[0]) for x,y in zip(true,preds) if not np.isnan(y)]
# mt=np.vstack(np.array(fre))
# mse = mean_squared_error(mt[0], mt[1])
# mae = mean_absolute_error(mt[0], mt[1])
# mape = mean_absolute_percentage_error(mt[0], mt[1])
# r2_score = r2_score(mt[0], mt[1])
# mse, mae, mape,r2_score


# # In[ ]:





# # In[62]:


# plt.figure(figsize=(20, 10))
# plt.plot(preds)
# plt.plot(true)
# plt.show()


# # In[84]:


# preds.squeeze().shape


# # In[85]:


# import pandas as pd
# import plotly.express as px

# # Sample data for two variables
# data = {
#     'Variable 1':true,
#     'Variable 2': preds.squeeze()
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Plot the data using plotly express
# fig = px.line(df, labels={"value": "Y Axis", "index": "Index"}, title="Two Variables Line Plot")

# # Show the plot
# fig.show()


# # In[63]:


# plt.plot(ttmp.loc[val_brdr+16:,'T'],ttmp.loc[val_brdr+16:,'Sdif'])


# # In[ ]:


# (preds-true).T**2


# # In[ ]:


# preds[true==np.nan]
# # predscta=preds[]


# # In[27]:


# ttmpct1=ttmpct.loc[ttmpct['T'].between('2021-12-27 00:00:00','2021-12-28 00:00:00')].copy()
# ttmpct1


# # In[6]:


# import joblib
# import seaborn as sns


# # In[51]:


# (preds,true)=joblib.load('vfr_delS_10T')
# ttmp=data.copy()
# ttmp=ttmp.rename({'Sdif':'True'},axis=1)
# import plotly.express as px
# ttmpct=ttmp.loc[val_brdr+16:].copy()
# ttmpct.loc[:,'pred']=preds
# ttmpct1=ttmpct.loc[ttmpct['T'].between('2021-12-27 00:00:00','2021-12-27 23:5:00')].copy()
# fig=px.line(ttmpct1,x='T',y=['True','pred'])
# fig.update_yaxes(title_text=r"$CO_{S2}-CO_{S1}$")
# font_size = 16
# fig.update_layout(
#     font=dict(size=font_size),
#     yaxis=dict(title_font=dict(size=font_size)),
#     legend=dict(
#         orientation="v",  # "h" for horizontal, "v" for vertical
#         x=0.5,              # Adjust x position (0 to 1)
#         y=1            # Adjust y position (smaller values move legend inside)
#     ))
# fig.update_xaxes(
#     tickformat='%H',  # Display only the hour part
#     type='date',      # Specify that x-axis values are datetime
#     title='Time (hrs)'      # Set x-axis title
# )



# # fig.show()
# import plotly.io as pio
# pio.write_image(fig, 'truepred.png', format='png')


# # In[26]:


# ttmp


# # In[45]:


# ttmpct.loc[ttmpct['T'].between('2021-11-15 01:00:00','2021-12-20 23:50:00')]


# # In[42]:


# # (preds,true)=joblib.load('vfr_delS_10T')
# ttmp=data.copy()
# # ttmp=ttmp.rename({'Sdif':'True'},axis=1)
# import plotly.express as px
# ttmpct=ttmp.loc[val_brdr+16:].copy()
# # ttmpct.loc[:,'pred']=preds
# ttmpct1=ttmpct.loc[ttmpct['T'].between('2021-12-19 01:00:00','2021-12-20 23:50:00')].copy()
# fig=px.line(ttmpct1,x='T',y=['Tbig_hist','S2A'])
# fig.update_yaxes(title_text=r"$CO_{S2}$")
# font_size = 16
# fig.update_layout(
#     font=dict(size=font_size),
#     yaxis=dict(title_font=dict(size=font_size)),
#     legend=dict(
#         orientation="h",  # "h" for horizontal, "v" for vertical
#         x=0.5,              # Adjust x position (0 to 1)
#         y=1            # Adjust y position (smaller values move legend inside)
#     ))
# fig.update_xaxes(
#     tickformat='%H',  # Display only the hour part
#     type='date',      # Specify that x-axis values are datetime
#     title='Time (hrs)'      # Set x-axis title
# )



# fig.show()
# # import plotly.io as pio
# # pio.write_image(fig, 'truepred.png', format='png')


# # In[36]:


# plt.plot(ttmp['T'],ttmp['Tbig_hist'])


# # In[ ]:


# import joblib
# (preds,true)=joblib.load('2var_15T')


# # In[ ]:


# # sns.lineplot(data=ttmpct, x='T', y='True', label='True')
# # sns.lineplot(data=ttmpct, x='T', y='pred', label='Pred')
# # plt.xlabel('time')
# # plt.ylabel(r"$\Delta CO$")

