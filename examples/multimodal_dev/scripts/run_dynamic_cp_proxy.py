"""Portable eight-GPU synthetic proxy recipe for baseline/static/joint Dynamic CP."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm',choices=('baseline','static','joint'),required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--train-iters',type=int,choices=(2,50),default=50)
parser.add_argument('--dry-run',action='store_true')
args=parser.parse_args()
source=Path(__file__).resolve().parents[3]
packet=Path(__file__).resolve().parent
arm=args.arm; phase='smoke' if args.train_iters==2 else 'formal'
audit=args.dry_run
out=args.output_dir.resolve()
assert int(os.environ.get('NNODES','2'))*int(os.environ.get('GPUS_PER_NODE','4'))==8
if not audit:
 assert os.environ.get('MASTER_ADDR'), 'Set MASTER_ADDR to the rendezvous host'
 assert os.environ.get('NODE_RANK') is not None, 'Set NODE_RANK separately on each node'
 from importlib.metadata import distribution, version
 from packaging.version import Version
 import hashlib
 assert Version(version('nvidia-resiliency-ext'))>=Version('0.6.0')
 te_cp=distribution('transformer_engine').locate_file(
  'transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py')
 assert hashlib.sha256(te_cp.read_bytes()).hexdigest()=='5f16ad06f60fd8bec3d21b56e07dcf027617528a0b195eb523a9982f212b6430', 'Apply the documented version-gated TE patch first'
env=os.environ.copy()
env.update(DRY_RUN='1',GPUS_PER_NODE=os.environ.get('GPUS_PER_NODE','4'),
 NNODES=os.environ.get('NNODES','2'),MODEL_VARIANT='proxy',
 VISION_NUM_LAYERS='1',NUM_LAYERS='4',NUM_EXPERTS='16',TP='1',EP='1',PP='1',
 CP='4',MBS='1',GBS='128',SEQ_LEN='2048',DATASET_PROVIDER='mdp_mock',
 TOKENIZER_TYPE='NullMultimodalTokenizer',VOCAB_SIZE='248320',MTP_NUM_LAYERS='0',
 SAVE_CHECKPOINT='0',USE_FSDP='0',RECOMPUTE='0',RECOMPUTE_VISION='1',
 USE_PACKED_SEQUENCE='1',TRAIN_ITERS='2' if phase=='smoke' else '50',
 FORCE_LOAD_BALANCING='1',ROOT_DIR=str(out/'unused_checkpoint')+'/',
 TENSORBOARD_LOGS_PATH=str(out/'tensorboard'),MEGATRON_LM_PATH=str(source),
 NCCL_NVLS_ENABLE='0',NVTE_FUSED_ATTN='1',NVTE_FLASH_ATTN='0',NVTE_UNFUSED_ATTN='0',
 PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',
 PYTHONPATH=str(source)+os.pathsep+os.environ.get("PYTHONPATH",""),ARM=arm)
text=subprocess.check_output(['bash',str(source/'examples/multimodal_dev/scripts/run_qwen35_vl.sh')],env=env,text=True)
lines=[x for x in text.splitlines() if x.startswith('torchrun ')]
assert len(lines)==1
argv=shlex.split(lines[0])
def replace(flag,value):
 assert argv.count(flag)==1,flag
 argv[argv.index(flag)+1]=str(value)
argv.remove('--sequence-parallel')
replace('--cp-comm-type','p2p')
replace('--cross-entropy-fusion-impl','native')
replace('--eval-iters',0)
replace('--kv-channels',128)
replace('--rotary-percent',0.5)
entry_index=argv.index(str(source/'examples/multimodal_dev/pretrain_multimodal.py'))
argv[entry_index:entry_index]=['--node_rank',os.environ.get('NODE_RANK','0')]
assert argv.index('--node_rank')<argv.index(str(source/'examples/multimodal_dev/pretrain_multimodal.py'))
argv+=['--attention-backend','fused','--seed','1234',
 '--dataloader-type','single','--num-workers','0',
 '--max-seqlen-per-dp-cp-rank','512']
if arm!='baseline':
 argv+=['--mdp-enable','--mdp-vision-capture-mode','source-pixel-sidecar',
        '--mdp-encoder-cp','4','--mdp-encoder-max-payload-rows',
        '26000' if arm=='joint' else '57600','--mdp-encoder-assignment-policy','lpt']
if arm=='joint':
 argv+=['--mdp-dynamic-encoder-cp','--mdp-min-dynamic-encoder-cp-size','1',
        '--dynamic-context-parallel','--min-dynamic-context-parallel-size','1']
assert '--recompute-granularity' not in argv and '--save' not in argv and '--load' not in argv
flags=[x for x in argv if x.startswith('--')]
assert len(flags)==len(set(flags)),[x for x in flags if flags.count(x)>1]
print('PROXY_REVIEWED_ARGV '+json.dumps(argv),flush=True)
if not audit:
 argv[argv.index(str(source/'examples/multimodal_dev/pretrain_multimodal.py'))]=str(packet/'observe_dynamic_cp_proxy.py')
 os.chdir(source)
 env['DRY_RUN']='0'
 os.execvpe(argv[0],argv,env)
