"""Probe the frozen neural actor initialization on fixed validation-prior states.

This tool reconstructs CPU initialization, never reads rewards or eval outcomes,
never updates parameters, and does not claim equality to a remote GPU trajectory.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def conditional_moments(actions):
    a=np.asarray(actions,dtype=np.float64)
    if a.ndim!=3 or a.shape[2]!=5 or a.shape[0]<2 or a.shape[1]<2 or not np.isfinite(a).all():
        raise ValueError('Expected finite states x repeated draws x five actions')
    if np.max(np.abs(a))>1+1e-6:raise ValueError('Illegal executed action')
    radius=np.linalg.norm(a[:,:,:3],axis=-1);means=radius.mean(axis=1);std=radius.std(axis=1,ddof=1)
    return dict(states=a.shape[0],draws_per_state=a.shape[1],radius_mean=float(means.mean()),
                radius_pooled_median=float(np.median(radius)),radius_per_state_mean=means.tolist(),
                radius_per_state_sample_std=std.tolist(),mean_within_state_radius_std=float(std.mean()),
                between_state_radius_mean_std=float(means.std(ddof=1)),
                mean_within_state_xyz_covariance_trace=float(a[:,:,:3].var(axis=1,ddof=1).sum(axis=-1).mean()),
                pooled_xyz_covariance_trace=float(a[:,:,:3].reshape(-1,3).var(axis=0,ddof=1).sum()),
                near_xyz_cube_boundary_fraction=float((np.max(np.abs(a[:,:,:3]),axis=-1)>=1-1e-6).mean()))


def compare_forward_maps(location,scale,noise,compose,transform=np.tanh,noise_scale=1.):
    """Change execution mapping while retaining all six actor heads and draws."""
    loc,std,eps=np.asarray(location),np.asarray(scale),np.asarray(noise)
    if loc.ndim!=2 or loc.shape[1]!=6 or loc.shape!=std.shape or eps.ndim!=3 or eps.shape[0]!=len(loc) or eps.shape[2]!=6:
        raise ValueError('Posthoc heads and noise panel shape mismatch')
    if not all(np.isfinite(x).all() for x in [loc,std,eps]) or np.any(std<=0) or not np.isfinite(noise_scale) or noise_scale<=0:
        raise ValueError('Invalid Gaussian parameters or noise')
    raw=loc[:,None,:]+noise_scale*std[:,None,:]*eps
    policy=np.asarray(transform(raw.reshape(-1,6))).reshape(raw.shape)
    error=float(np.max(np.abs(policy-np.tanh(raw))))
    if error>2e-6:raise ValueError('This audit requires the legacy all-tanh policy')
    post=np.asarray(compose(policy.reshape(-1,6))).reshape(*policy.shape[:2],5)
    cart=policy[:,:,[0,1,2,4,5]]
    np.testing.assert_array_equal(post[:,:,3:],cart[:,:,3:])
    u=policy[:,:,:3];r=np.linalg.norm(u,axis=-1,keepdims=True)
    preclip=np.exp(policy[:,:,3:4])*u/np.maximum(r,1e-6)
    post_m,cart_m=conditional_moments(post),conditional_moments(cart)
    difference=np.asarray(post_m['radius_per_state_mean'])-np.asarray(cart_m['radius_per_state_mean'])
    return dict(noise_scale=noise_scale,posthoc=post_m,same_posthoc_heads_cartesian_execution=cart_m,
                mean_radius_difference=float(difference.mean()),positive_state_differences=int((difference>0).sum()),
                scalar_execution_max_difference=0.,actual_xyz_clipping_fraction=float((np.max(np.abs(preclip),axis=-1)>1).mean()),
                normalization_floor_fraction=float((r[:,:,0]<1e-6).mean()),policy_transform_check_max_error=error,
                raw_speed_location_per_state=loc[:,3].tolist(),raw_speed_scale_per_state=std[:,3].tolist(),
                scope='Within one frozen actor: same Gaussian heads and draws, two execution maps. Counterfactual Cartesian execution discards the speed head; it is not a separately trained baseline.')


def verify_source(root,manifest):
    for row in manifest['python_sources']:
        if digest(root/row['path'])!=row['sha256']:raise ValueError('Frozen source mismatch: '+row['path'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source-root','release-manifest','inputs','dataset','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--dataset-sha256',required=True)
    parser.add_argument('--states',type=int,default=128);parser.add_argument('--draws',type=int,default=64)
    args=parser.parse_args()
    if args.states<2 or args.draws<2:raise ValueError('At least two states and draws required')
    manifest=json.loads(args.release_manifest.read_text());verify_source(args.source_root,manifest)
    inputs_sha=digest(args.inputs);inputs=json.loads(args.inputs.read_text())
    if digest(args.dataset)!=args.dataset_sha256:raise ValueError('Validation-prior hash mismatch')
    with np.load(args.dataset) as data:
        observations=data['observations'];eligible=np.flatnonzero(data['terminals']==0)
        chosen=np.random.default_rng(20261020).choice(eligible,size=args.states,replace=False)
        panel=np.asarray(observations[chosen],dtype=np.float32)
    if panel.ndim!=2 or not np.isfinite(panel).all():raise ValueError('Invalid observation panel')
    noise=np.random.default_rng(20261023).standard_normal((args.states,args.draws,6)).astype('float32')
    sys.path.insert(0,str(args.source_root.resolve()))
    import jax
    import jax.numpy as jnp
    import flax
    from agents.acrlpd import ACRLPDAgent
    from direction_speed.adapter import DirectionSpeedAdapter
    if jax.default_backend()!='cpu':raise ValueError('Run this initialization audit with JAX_PLATFORMS=cpu')
    for name in ['agents.acrlpd','direction_speed.adapter','rlpd_distributions.parameter_heads','utils.flax_utils']:
        Path(sys.modules[name].__file__).resolve().relative_to(args.source_root.resolve())
    adapter=DirectionSpeedAdapter(speed_squash='tanh',speed_interpretation='log',speed_bound='sphere')
    rows=[]
    for seed in [10,11,12,13]:
        for method,group in [('native','aa_triple2_native_20261019'),('posthoc','aa_triple2_legacy_latent_20261019')]:
            config=copy.deepcopy(inputs[group]['effective_config'])
            original_bytes=json.dumps(config,indent=2,sort_keys=True).encode()
            if hashlib.sha256(original_bytes).hexdigest()!=inputs[group]['effective_config_sha256']:
                raise ValueError('Captured effective configuration changed')
            if config['horizon_length']!=1 or config['env_action_dim']!=5:raise ValueError('Requires the frozen H1 five-dimensional recipe')
            example=np.zeros(5,dtype=np.float32) if method=='native' else adapter.decompose(np.zeros(5,dtype=np.float32))
            agent=ACRLPDAgent.create(seed,panel[0],example,config)
            before=np.asarray(agent.rng).copy();initial_step=int(agent.network.step)
            if initial_step!=1:raise ValueError('Expected fresh project TrainState')
            dist=agent.network.select('actor')(jnp.asarray(panel))
            loc=np.asarray(dist.raw_distribution.mean());std=np.asarray(dist.raw_distribution.stddev())
            actor_leaves=jax.tree_util.tree_leaves(agent.network.params['modules_actor'])
            actor_sha=hashlib.sha256(b''.join(np.asarray(v).tobytes() for v in actor_leaves)).hexdigest()
            transform=lambda raw:np.asarray(dist.raw_to_policy(jnp.asarray(raw)))
            record=dict(seed=seed,method=method,optimizer_updates=0,actor_parameter_sha256=actor_sha,
                        effective_config_sha256=inputs[group]['effective_config_sha256'],cases=[])
            for scale in [.2,.5,1.,2.]:
                if method=='posthoc':record['cases'].append(compare_forward_maps(loc,std,noise,adapter.compose,transform,scale))
                else:
                    raw=loc[:,None,:]+scale*std[:,None,:]*noise[:,:,[0,1,2,4,5]]
                    executed=transform(raw.reshape(-1,5)).reshape(args.states,args.draws,5)
                    record['cases'].append(dict(noise_scale=scale,native=conditional_moments(executed)))
            np.testing.assert_array_equal(agent.rng,before)
            if int(agent.network.step)!=initial_step:raise ValueError('Probe updated the model')
            rows.append(record);print(json.dumps(dict(seed=seed,method=method,finished=True)),flush=True)
    verify_source(args.source_root,manifest)
    if digest(args.inputs)!=inputs_sha:raise ValueError('Input configs changed during analysis')
    report=dict(status='complete',scope='Frozen CPU neural initialization on fixed validation-prior states; no environment interactions, training updates, eval outcomes or restored trained weights.',
                training_source_commit=manifest['source_commit'],source_files_verified=len(manifest['python_sources']),
                source_root=str(args.source_root),inputs_sha256=inputs_sha,dataset_sha256=args.dataset_sha256,
                panel=dict(selection_seed=20261020,count=args.states,indices=chosen.tolist(),
                           sha256=hashlib.sha256(panel.tobytes()+chosen.astype('int64').tobytes()).hexdigest()),
                noise=dict(seed=20261023,draws_per_state=args.draws,sha256=hashlib.sha256(noise.tobytes()).hexdigest()),
                backend=jax.default_backend(),jax=jax.__version__,flax=flax.__version__,numpy=np.__version__,
                tool_sha256=digest(Path(__file__)),records=rows,
                limitations=['CPU reconstruction does not certify bitwise equality to remote GPU initialization.',
                             'Changing noise scale holds Gaussian locations and scales fixed; trained actors can adapt both.',
                             'The same-head Cartesian counterfactual isolates execution mapping and is not a trained baseline.',
                             'Conditional noise covariance is separated from variation across prior states; neither establishes task benefit.',
                             'No causal attribution of the historical performance gains or confirmatory claim from four initialization seeds.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',models=len(rows),output=str(args.output))))


if __name__=='__main__':main()
