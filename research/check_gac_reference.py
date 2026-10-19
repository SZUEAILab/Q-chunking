"""Compare the local GAC learner with classes and update statements from pinned AST.

The official training entry is not run. No generated surrogate update is used:
its original learning-starts branch executes with a fixed synthetic replay stub.
"""
import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agents.gac_reference import GACReferenceAgent, GACReferenceConfig, UPSTREAM_COMMIT


def load_upstream(root):
    repository = root if (root / '.git').exists() else root / 'repository'
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()!=UPSTREAM_COMMIT:
        raise ValueError('Unexpected official GAC revision')
    path = root / 'GAC.py'
    for filename in ('GAC.py', 'LICENSE'):
        exact = subprocess.check_output(['git','show',UPSTREAM_COMMIT+':'+filename],cwd=repository)
        if (root/filename).read_bytes() != exact:
            raise ValueError('Official source differs from its pinned git object: '+filename)
    tree=ast.parse(path.read_text())
    definitions=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in ('Args','SoftQNetwork','GACActor')]
    if len(definitions)!=3:raise ValueError('Official class set differs')
    module=ModuleType('_pinned_gac_oracle');sys.modules[module.__name__]=module
    module.__dict__.update(dict(dataclass=dataclass,np=np,torch=torch,nn=nn,F=F))
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(path),'exec'),module.__dict__)
    main=next(node for node in tree.body if isinstance(node,ast.If) and ast.unparse(node.test)=="__name__ == '__main__'")
    loop=next(node for node in main.body if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='global_step')
    candidates=[node for node in loop.body if isinstance(node,ast.If) and ast.unparse(node.test)=='global_step > args.learning_starts']
    if len(candidates)!=1:raise ValueError('Official learner branch differs')
    code=compile(ast.Module(body=candidates,type_ignores=[]),str(path),'exec')
    return module,code,hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a,b):
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):raise AssertionError('State dictionary keys differ')
        return max((compare(a[key],b[key]) for key in a),default=0.)
    if isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):raise AssertionError('State sequence differs')
        return max((compare(x,y) for x,y in zip(a,b)),default=0.)
    if torch.is_tensor(a):a=a.detach().cpu().numpy()
    if torch.is_tensor(b):b=b.detach().cpu().numpy()
    a,b=np.asarray(a),np.asarray(b)
    np.testing.assert_array_equal(a,b)
    if np.issubdtype(a.dtype,np.number):return float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
    return 0.


def run_case(official,code,config,action_dim,device="cpu",observation_dim=7):
    device=torch.device(device)
    cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    local=GACReferenceAgent(20261031,observation_dim,-np.ones(action_dim),np.ones(action_dim),config,device)
    args=official.Args()
    for name,value in asdict(config).items():setattr(args,name,value)
    env=SimpleNamespace(single_observation_space=SimpleNamespace(shape=(observation_dim,)),single_action_space=SimpleNamespace(shape=(action_dim,),low=-np.ones(action_dim,dtype='float32'),high=np.ones(action_dim,dtype='float32')))
    context=dict(torch=torch,F=F,args=args)
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(20261031)
        context['actor']=official.GACActor(env,args).to(device)
        for name in ('qf1','qf2','qf1_target','qf2_target'):context[name]=official.SoftQNetwork(env).to(device)
        for name in ('qf1','qf2'):context[name+'_target'].load_state_dict(context[name].state_dict())
        context['q_optimizer']=torch.optim.Adam(list(context['qf1'].parameters())+list(context['qf2'].parameters()),lr=args.q_lr)
        context['actor_optimizer']=torch.optim.Adam(list(context['actor'].parameters()),lr=args.policy_lr)
        reference_rng=torch.get_rng_state().clone()
        reference_cuda=torch.cuda.get_rng_state(cuda_devices[0]).clone() if cuda_devices else None
    def states():
        errors=[compare(getattr(local,name).state_dict(),context[name].state_dict()) for name in ('actor','qf1','qf2','qf1_target','qf2_target')]
        errors += [compare(local.q_optimizer.state_dict(),context['q_optimizer'].state_dict()),compare(local.actor_optimizer.state_dict(),context['actor_optimizer'].state_dict()),compare(local.rng_cpu,reference_rng)]
        if cuda_devices:errors.append(compare(local.rng_cuda,reference_cuda))
        return max(errors)
    errors=[states()];rng=np.random.default_rng(20261031)
    actor_updates=target_updates=0
    for step in range(config.learning_starts+1,config.learning_starts+7):
        observation=rng.normal(size=observation_dim).astype('float32')
        sampled=local.sample_actions(observation)
        with torch.random.fork_rng(devices=cuda_devices),torch.no_grad():
            torch.set_rng_state(reference_rng)
            if cuda_devices:torch.cuda.set_rng_state(reference_cuda,cuda_devices[0])
            expected=context['actor'].get_action(torch.as_tensor(observation[None],device=device))[0][0]
            reference_rng=torch.get_rng_state().clone()
            if cuda_devices:reference_cuda=torch.cuda.get_rng_state(cuda_devices[0]).clone()
        compare(sampled,expected);errors.append(states())
        size=config.batch_size
        batch=dict(observations=rng.normal(size=(size,observation_dim)).astype('float32'),next_observations=rng.normal(size=(size,observation_dim)).astype('float32'),
                   actions=rng.uniform(-.8,.8,(size,action_dim)).astype('float32'),rewards=np.resize(np.asarray([-1,0,1,-2,0,-1,0,1],np.float32),size)[:,None],
                   dones=np.resize(np.asarray([0,0,1,0,1,0,1,1],np.float32),size)[:,None])
        info=local.update(batch,step)
        data=SimpleNamespace(**{name:torch.as_tensor(value,device=device) for name,value in batch.items()})
        context.update(rb=SimpleNamespace(sample=lambda size:data),global_step=step)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.set_rng_state(reference_rng)
            if cuda_devices:torch.cuda.set_rng_state(reference_cuda,cuda_devices[0])
            exec(code,context)
            reference_rng=torch.get_rng_state().clone()
            if cuda_devices:reference_cuda=torch.cuda.get_rng_state(cuda_devices[0]).clone()
        errors.append(states())
        for name in ('qf_loss','qf1_loss','qf2_loss'):
            compare(info[name],float(context[name].detach()))
        if step%config.policy_frequency==0:
            actor_updates+=config.policy_frequency
            for name in ('actor_loss','sac_actor_loss','kappa_reg_loss'):compare(info[name],float(context[name].detach()))
        if step%config.target_network_frequency==0:target_updates+=1
    assert local.critic_updates==6 and local.actor_updates==actor_updates and local.target_updates==target_updates
    return dict(status='passed',config=asdict(config),action_dim=action_dim,observation_dim=observation_dim,device=str(device),updates=6,interleaved_collection_samples=6,
                critic_updates=local.critic_updates,actor_updates=actor_updates,target_updates=target_updates,max_absolute_state_error=max(errors))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--upstream-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');parser.add_argument('--observation-dim',type=int,default=7)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing oracle evidence')
    torch.set_num_threads(1)
    learner=Path(sys.modules[GACReferenceAgent.__module__].__file__);before=hashlib.sha256(learner.read_bytes()).hexdigest()
    official,code,source_sha=load_upstream(args.upstream_root)
    if args.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):raise ValueError('Expose one CUDA GPU for the numerical check')
    cases=[run_case(official,code,GACReferenceConfig(learning_starts=0),5,args.device,args.observation_dim),
           run_case(official,code,GACReferenceConfig(learning_starts=2,policy_frequency=1,target_network_frequency=2),5,args.device,args.observation_dim),
           run_case(official,code,GACReferenceConfig(learning_starts=0,policy_frequency=3,target_network_frequency=2,alpha=.1,radius=.5),3,args.device,args.observation_dim)]
    assert hashlib.sha256(learner.read_bytes()).hexdigest()==before
    assert hashlib.sha256((args.upstream_root/'GAC.py').read_bytes()).hexdigest()==source_sha
    report=dict(status='passed',upstream_commit=UPSTREAM_COMMIT,upstream_source_sha256=source_sha,local_source_sha256=before,
                oracle_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch=torch.__version__,backend=args.device,cases=cases,
                scope='Official network initialization, interleaved collection RNG, complete learner updates/optimizers/targets and losses. No environment/vector-budget or task-performance equivalence claim.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],cases=report['cases'])),flush=True)


if __name__=='__main__':main()
