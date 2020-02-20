# NNScheduler
workflow_launch - main file with experiment scenario  
actor - parameters of Q-learning agent  
wf_gen_funcs - functions for building workflow's structure tree and random workflows  
env.context - desription of workflow scheduling problem, generation of state  
env.entities - nodes, tasks and performance models  

# Run Example
First you should load server-side with model using followed command with parameters
```
python server.py --actor-type=fc 
```

Second you should start client-side using command with parameters

```
python episode.py --num-episodes=1000 --wfs-name=Montage_100
```
