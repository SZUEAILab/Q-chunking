"""Read-only posthoc geometry probes; no trained-performance or causal claim."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from direction_speed.adapter import DirectionSpeedAdapter, EPS


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def legacy_adapter():
    return DirectionSpeedAdapter(speed_squash='tanh',speed_interpretation='log',speed_bound='sphere')


def radius_stats(actions):
    actions=np.asarray(actions,dtype=np.float64)
    if actions.ndim!=2 or actions.shape[1]!=5 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError('Expected finite nonempty five-dimensional actions')
    r=np.linalg.norm(actions[:,:3],axis=-1)
    return dict(count=len(actions),mean=float(r.mean()),median=float(np.median(r)),
                quantiles=np.quantile(r,[0,.1,.5,.9,1]).tolist(),
                below_regular_branch_floor_fraction=float((r<np.exp(-1)-1e-7).mean()),
                outside_unit_sphere_fraction=float((r>1+1e-6).mean()),
                near_xyz_cube_boundary_fraction=float((np.abs(actions[:,:3]).max(axis=-1)>=1-1e-6).mean()))


def probe(raw,scale):
    """Common zero-mean Gaussian coordinates isolate the forward map only."""
    raw=np.asarray(raw,dtype=np.float64)
    if raw.ndim!=2 or raw.shape[1]!=6 or not len(raw) or not np.isfinite(raw).all() or not np.isfinite(scale) or scale<=0:
        raise ValueError('Invalid probe panel or scale')
    policy=np.tanh(raw*scale)
    native=policy[:,[0,1,2,4,5]]
    u=policy[:,:3];norm=np.linalg.norm(u,axis=-1,keepdims=True)
    direction=u/np.maximum(norm,EPS)
    preclip=np.exp(policy[:,3:4])*direction
    posthoc=legacy_adapter().compose(policy)
    normalized_native=native[:,:3]/np.maximum(np.linalg.norm(native[:,:3],axis=-1,keepdims=True),EPS)
    expected=np.clip(np.concatenate([preclip,policy[:,4:]],axis=-1),-1,1)
    np.testing.assert_allclose(posthoc,expected,rtol=0,atol=1e-12)
    return dict(scale=float(scale),native=radius_stats(native),posthoc=radius_stats(posthoc),
                actual_preclip_xyz_clipping_fraction=float((np.abs(preclip).max(axis=-1)>1).mean()),
                normalization_floor_fraction=float((norm[:,0]<EPS).mean()),
                shared_preclip_direction_max_difference=float(np.max(np.abs(direction-normalized_native))),
                posthoc_radius=np.linalg.norm(posthoc[:,:3],axis=-1),
                native_radius=np.linalg.norm(native[:,:3],axis=-1))


def replay_geometry(actions,old_mapper=None):
    a=np.asarray(actions)
    stats=radius_stats(a)
    if np.any(np.abs(a)>1+1e-6):raise ValueError('Prior action exceeds normalized cube')
    encoded=legacy_adapter().decompose(a)
    magnitude=np.linalg.norm(a[:,:3],axis=-1)
    count=int(np.sum(encoded[:,3]<-1))
    result=dict(**stats,log_speed_below_actor_coordinate_bound_count=count,
                log_speed_below_actor_coordinate_bound_fraction=count/len(a),
                encoded_log_speed_quantiles=np.quantile(encoded[:,3],[0,.1,.5,.9,1]).tolist(),
                normalization_floor_count=int(np.sum(magnitude<EPS)),
                scope='Validation-prior action encoding only; not learned-policy occupancy or performance attribution. The latent replay speed coordinate may lie below -1 even though the full execution map has a near-zero-direction escape branch.')
    if old_mapper is not None:
        previous=old_mapper.decompose(a)
        result['june_vs_current_decompose_max_difference']=float(np.max(np.abs(encoded-previous)))
        result['june_vs_current_decompose_mismatched_rows']=int(np.sum(np.max(np.abs(encoded-previous),axis=-1)>1e-6))
        rng=np.random.default_rng(20261022);p=np.tanh(rng.normal(size=(1024,6)))
        result['june_vs_current_compose_max_difference']=float(np.max(np.abs(legacy_adapter().compose(p)-old_mapper.compose(p))))
    return result


def jacobian_check():
    """Compare independent calculus and finite differences with actual JAX map."""
    import jax
    import jax.numpy as jnp
    adapter=legacy_adapter()
    p=np.array([.2,.3,.4,-.25],dtype=np.float32)
    def function(v):
        return adapter.compose_jax(jnp.concatenate([v,jnp.array([.1,-.2])]))[:3]
    jac=np.asarray(jax.jacfwd(function)(jnp.asarray(p)))
    u=p[:3].astype(np.float64);r=np.linalg.norm(u);d=u/r;s=np.exp(float(p[3]))
    expected=np.column_stack([s/r*(np.eye(3)-np.outer(d,d)),s*d])
    delta=1e-3
    finite=np.column_stack([(np.asarray(function(jnp.asarray(p+delta*np.eye(4)[i])))-np.asarray(function(jnp.asarray(p-delta*np.eye(4)[i]))))/(2*delta) for i in range(4)])
    np.testing.assert_allclose(jac,expected,atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(jac,finite,atol=6e-5,rtol=6e-4)
    return dict(policy_point=p.tolist(),analytic_max_error=float(np.max(np.abs(jac-expected))),
                finite_difference_max_error=float(np.max(np.abs(jac-finite))),
                radial_direction_null_residual=float(np.linalg.norm(jac[:,:3]@u)),
                speed_gradient_action_residual=float(np.linalg.norm(jac[:,3]-s*d)),
                scope='One smooth, unclipped, regular-normalization point; derivative is with respect to post-squash policy coordinates, not all raw network outputs or a learned Q.')


def normalization_boundary(old_mapper=None):
    p=np.array([[0,0,0,0,0,0],[EPS/10,0,0,0,0,0]],dtype=np.float64)
    executed=legacy_adapter().compose(p)
    assert np.linalg.norm(executed[0,:3])==0
    assert np.isclose(np.linalg.norm(executed[1,:3]),.1)
    a=np.zeros((1,5),dtype=np.float64)
    result=dict(epsilon=EPS,zero_direction_execution=executed[0].tolist(),
                subepsilon_direction_execution=executed[1].tolist(),
                regular_branch_lower_radius=float(np.exp(-1)),
                current_zero_action_encoding=legacy_adapter().decompose(a)[0].tolist(),
                scope='The radius floor is conditional on direction norm >= epsilon. It is not a global support exclusion.')
    if old_mapper is not None:result['june_zero_action_encoding']=old_mapper.decompose(a)[0].tolist()
    return result


def plot_report(report,distributions,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13.4,4),layout='constrained')
    for name,label,color in [('native_radius','Cartesian native','#537ba0'),('posthoc_radius','Legacy posthoc','#ca6b40')]:
        values=np.sort(distributions[name]);indices=np.linspace(0,len(values)-1,min(2000,len(values)),dtype=int)
        axes[0].plot(values[indices],(indices+1)/len(values),label=label,color=color)
    axes[0].axvline(np.exp(-1),color='gray',ls=':',lw=1)
    axes[0].set(xlabel='Executed XYZ norm',ylabel='Empirical CDF',title='Common Gaussian probe, SD = 1')
    axes[0].legend(fontsize=8)
    for name,label,color in [('native','Cartesian native','#537ba0'),('posthoc','Legacy posthoc','#ca6b40')]:
        axes[1].plot([p['scale'] for p in report['probes']],[p[name]['median'] for p in report['probes']],'-o',label=label,color=color)
    axes[1].set(xlabel='Raw Gaussian standard deviation',ylabel='Median executed XYZ norm',title='Zero means; common input draws')
    names=list(report['validation_priors']);values=[100*report['validation_priors'][n]['log_speed_below_actor_coordinate_bound_fraction'] for n in names]
    axes[2].bar([n.replace('cube-','Cube ').title() for n in names],values,color='#7c9885')
    for index,v in enumerate(values):axes[2].text(index,v+.5,f'{v:.2f}%',ha='center',fontsize=9)
    axes[2].set(ylabel='Prior actions with log radius < -1 (%)',ylim=(0,max(values)+7),title='Latent replay / actor-coordinate mismatch')
    for ax in axes:ax.grid(alpha=.2,axis='y')
    fig.suptitle('Forward-map and replay diagnostics; no training or performance attribution',fontsize=11)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--data-manifest',type=Path,required=True)
    parser.add_argument('--release-manifest',type=Path,required=True)
    parser.add_argument('--legacy-source',type=Path,required=True)
    parser.add_argument('--legacy-source-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--figure',type=Path)
    parser.add_argument('--samples',type=int,default=200000)
    args=parser.parse_args()
    if args.samples<1000:raise ValueError('Use at least 1000 common probe draws')
    manifest=json.loads(args.release_manifest.read_text());root=Path(__file__).resolve().parents[1]
    paths=['direction_speed/adapter.py','direction_speed/speed.py','direction_speed/specs.py']
    expected={r['path']:r['sha256'] for r in manifest['python_sources']}
    for p in paths:
        if digest(root/p)!=expected[p]:raise ValueError('Mapper changed from frozen release: '+p)
    if digest(args.legacy_source)!=args.legacy_source_sha256:raise ValueError('Legacy source hash mismatch')
    spec=importlib.util.spec_from_file_location('june_posthoc_snapshot',args.legacy_source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    datasets={}
    for item in json.loads(args.data_manifest.read_text()):
        if not item['relative_path'].endswith('-val.npz'):continue
        p=args.dataset_root/item['relative_path']
        if p.stat().st_size!=item['size'] or digest(p)!=item['sha256']:raise ValueError('Prior file mismatch: '+str(p))
        with np.load(p) as f:actions=f['actions']
        name=p.name.split('-play-')[0];datasets[name]=dict(replay_geometry(actions,old),dataset_sha256=item['sha256'],dataset_path=str(p))
    if len(datasets)!=2:raise ValueError('Expected the two frozen validation priors')
    raw=np.random.default_rng(20261022).normal(size=(args.samples,6));probes=[];selected=None
    for scale in [.2,.5,1.,2.]:
        result=probe(raw,scale)
        arrays={name:result.pop(name) for name in ['posthoc_radius','native_radius']}
        if scale==1.:selected=arrays
        probes.append(result)
    report=dict(status='diagnostics_complete',analysis='Forward-map controlled probes and validation-prior encoding audit; no trained-performance attribution',
                source_commit=manifest['source_commit'],mapper_files={p:digest(root/p) for p in paths},
                tool_sha256=digest(Path(__file__)),legacy_source_sha256=digest(args.legacy_source),
                legacy_source_role='June snapshot for mapping comparisons; historical executed source remains uncertified',
                data_manifest_sha256=digest(args.data_manifest),release_manifest_sha256=digest(args.release_manifest),
                seed=20261022,samples=args.samples,raw_panel_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                probes=probes,validation_priors=datasets,jacobian=jacobian_check(),normalization_boundary=normalization_boundary(old),
                limitations=['Common Gaussian probes are not measurements of initial or trained neural policies.',
                             'The norm floor excludes the subepsilon direction branch; small executed actions remain possible there.',
                             'Replay-coordinate mismatch does not identify whether it helps or hurts performance.',
                             'The final effects of actor geometry, critic representation, entropy and clipping require controlled training contrasts.'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.figure:plot_report(report,selected,args.figure)
    print(json.dumps(dict(status=report['status'],prior_counts={n:r['count'] for n,r in datasets.items()},output=str(args.output))))


if __name__=='__main__':main()
