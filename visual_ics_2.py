# Supply-Chain Team Play – **rev-C2** (DQN bug-fix)
# =============================================================
# 2025-05-31 ++
#  🔧 Patch notes
#  • Fix PyTorch RuntimeError: subtraction with bool tensor.
#    - Cast `done` flag to float before 1-minus operation.
#  • ReplayBuffer now converts numpy → torch with explicit dtypes.
#  • Minor tidy-ups (device forwarding, progress print).

import random, os, numpy as np, matplotlib.pyplot as plt
from collections import deque
import torch, torch.nn as nn, torch.nn.functional as F


# ------------------------------------------------------------
# 1. Environment (unchanged from rev-C)
# ------------------------------------------------------------
class SupplyChainTeamEnv:
    """4-stage supply chain with cyber attacks + shaping."""
    def __init__(self, grid_size=12, n_agents=2, hack_p=0.10,
                 max_steps=250, shaping_gamma=0.99):
        self.n_agents=n_agents; self.grid_size=grid_size
        self.raw=(0,0); self.factory=(grid_size//2, grid_size//2)
        self.qc=(grid_size//2, grid_size-3); self.warehouse=(grid_size-1,grid_size-1)
        self.hack_p=hack_p; self.max_steps=max_steps; self.shaping_gamma=shaping_gamma
        self.reset()

    def _clip(self,v): return max(0,min(v,self.grid_size-1))
    def _target(self,s): return [self.raw,self.factory,self.qc,self.warehouse][s]
    def _phi(self,i):
        x,y=self.pos[i]; tx,ty=self._target(self.item[i])
        return -(abs(x-tx)+abs(y-ty))
    def _state(self,i):
        x,y=self.pos[i]
        return np.array([x/11,y/11,self.item[i]/3,min(self.trust[i],2)/2],dtype=np.float32)

    def reset(self):
        self.pos=[]; self.item=[]; self.speed=[]; self.done=[]; self.trust=[]
        for _ in range(self.n_agents):
            self.pos.append([random.randint(0,11),random.randint(0,11)])
            self.item.append(0); self.speed.append(1); self.done.append(False); self.trust.append(0)
        self.t=0
        return [self._state(i) for i in range(self.n_agents)]

    def step(self,acts):
        self.t+=1; local=[0.0]*self.n_agents; phi_old=[self._phi(i) for i in range(self.n_agents)]
        for i in range(self.n_agents):
            if self.done[i]: continue
            dx=dy=0
            if acts[i]==0: dy=1
            elif acts[i]==1: dy=-1
            elif acts[i]==2: dx=-1
            elif acts[i]==3: dx=1
            self.pos[i][0]=self._clip(self.pos[i][0]+dx*self.speed[i])
            self.pos[i][1]=self._clip(self.pos[i][1]+dy*self.speed[i])
            local[i]-=1
            x,y=self.pos[i]
            if self.item[i]==0 and (x,y)==self.raw:
                self.item[i]=1; local[i]+=5
            elif self.item[i]==1 and (x,y)==self.factory:
                self.item[i]=2; local[i]+=10
            elif self.item[i]==2 and (x,y)==self.qc:
                self.item[i]=3; local[i]+=10
            elif self.item[i]==3 and (x,y)==self.warehouse:
                local[i]+=150; self.done[i]=True
            if (not self.done[i]) and random.random()<self.hack_p:
                atk=random.choice(["speed","tp","dmg"])
                if atk=="speed": self.speed[i]=random.choice([0,2,3])
                elif atk=="tp": self.pos[i]=[random.randint(0,11),random.randint(0,11)]
                elif atk=="dmg" and self.item[i]>=1: self.item[i]=max(1,self.item[i]-1)
                if random.random()<0.5: self.trust[i]=min(self.trust[i]+1,2); local[i]+=1
                else: local[i]-=2
        for i in range(self.n_agents):
            phi=self._phi(i); local[i]+=self.shaping_gamma*phi-phi_old[i]
        return [self._state(i) for i in range(self.n_agents)], sum(local), all(self.done) or self.t>=self.max_steps


# ------------------------------------------------------------
# 2. Replay Buffer
# ------------------------------------------------------------
class ReplayBuffer:
    def __init__(self,cap):
        self.cap=cap; self.buf=deque(maxlen=cap)
    def push(self,*exp):
        self.buf.append(exp)
    def sample(self,batch):
        batch=random.sample(self.buf,batch)
        s,a,r,ns,d=map(np.array,zip(*batch))
        return (torch.tensor(s,dtype=torch.float32),
                torch.tensor(a,dtype=torch.long),
                torch.tensor(r,dtype=torch.float32),
                torch.tensor(ns,dtype=torch.float32),
                torch.tensor(d,dtype=torch.float32))
    def __len__(self): return len(self.buf)
    def save(self,path): np.savez_compressed(path,arr=list(self.buf))
    def load(self,path): self.buf=deque(list(np.load(path,allow_pickle=True)["arr"]),maxlen=self.cap)


# ------------------------------------------------------------
# 3. DQN Agent
# ------------------------------------------------------------
class Net(nn.Module):
    def __init__(self,inp=4,out=5):
        super().__init__()
        self.fc1=nn.Linear(inp,64); self.fc2=nn.Linear(64,64); self.out=nn.Linear(64,out)
    def forward(self,x):
        x=F.relu(self.fc1(x)); x=F.relu(self.fc2(x)); return self.out(x)

class DQNAgent:
    def __init__(self,lr=1e-3,gamma=0.99,n_actions=5,device="cpu"):
        self.net=Net(out=n_actions).to(device); self.target=Net(out=n_actions).to(device)
        self.target.load_state_dict(self.net.state_dict())
        self.opt=torch.optim.Adam(self.net.parameters(),lr=lr)
        self.gamma=gamma; self.device=device
    def act(self,state,eps):
        if random.random()<eps: return random.randint(0,4)
        with torch.no_grad():
            q=self.net(torch.tensor(state,dtype=torch.float32,device=self.device))
            return int(torch.argmax(q).item())
    def learn(self,batch):
        s,a,r,ns,d=batch
        s,ns,r,d=s.to(self.device),ns.to(self.device),r.to(self.device),d.to(self.device)
        q_sa=self.net(s).gather(1,a.unsqueeze(1)).squeeze(1)
        q_next=self.target(ns).max(1)[0]
        target=r + self.gamma * q_next * (1 - d)  # d is float 0/1
        loss=F.mse_loss(q_sa,target.detach())
        self.opt.zero_grad(); loss.backward(); self.opt.step()
    def update_target(self): self.target.load_state_dict(self.net.state_dict())


# ------------------------------------------------------------
# 4. Training loop
# ------------------------------------------------------------

def train_dqn(episodes=400,buffer_cap=50000,start_learn=1000,batch=64,
              target_interval=500,save_buffer=None,load_buffer=None):
    env=SupplyChainTeamEnv(); buf=ReplayBuffer(buffer_cap)
    if load_buffer and os.path.exists(load_buffer): buf.load(load_buffer); print("Loaded buffer",len(buf))
    ag=DQNAgent(device="cpu")
    eps,eps_end,eps_decay=1.0,0.05,0.0005
    total_steps=0; R_hist=[]; S_hist=[]
    for ep in range(episodes):
        s=env.reset(); ep_R=0; done=False
        while not done:
            acts=[ag.act(s[i],eps) for i in range(env.n_agents)]
            ns,g,done=env.step(acts)
            buf.push(s[0],acts[0],g,ns[0],float(done))
            s=ns; ep_R+=g; total_steps+=1; eps=max(eps_end,eps-eps_decay)
            if len(buf)>=start_learn and total_steps%4==0:
                ag.learn(buf.sample(batch))
            if total_steps%target_interval==0: ag.update_target()
        R_hist.append(ep_R); S_hist.append(all(env.done))
        if ep%50==0:
            print(f"Ep {ep:3d} | Ret {ep_R:6.1f} | succ {S_hist[-1]} | eps {eps:.2f} | buf {len(buf)}")
    if save_buffer: buf.save(save_buffer)
    return ag,R_hist,S_hist


# ------------------------------------------------------------
# 5. Quick demo
# ------------------------------------------------------------
if __name__=="__main__":
    agent,R,S=train_dqn()
    plt.plot(R); plt.title("Return"); plt.grid(); plt.show()
    sm=[np.mean(S[max(0,i-20):i+1]) for i in range(len(S))]
    plt.plot(sm); plt.title("Success rate (win20)"); plt.ylim(0,1); plt.grid(); plt.show()
