"""Smoke-only identity and first native dynamic-plan observation."""
import collections
import hashlib
import json
import os
import runpy
from pathlib import Path
import sys
from examples.multimodal_dev.data.mdp_mock import MdpThdMockDataset
from megatron.core.mdp import dynamic_cp_plan, dynamic_cp_d4_encoder_capture

original=MdpThdMockDataset.__getitem__
records=[]
source_info=[]
def observed(self,index):
 value=original(self,index)
 if len(records)<64:
  grids,chunks=self.scenarios[index%len(self.scenarios)]
  records.append((int(index),grids,chunks))
  if len(records)==64:
   print('PROXY_SAMPLE_WINDOW '+json.dumps(dict(rank=os.environ['RANK'],
    count=64,ids=[x[0] for x in records],
    signature=hashlib.sha256(json.dumps(records).encode()).hexdigest())),flush=True)
   MdpThdMockDataset.__getitem__=original
 return value
MdpThdMockDataset.__getitem__=observed
if os.environ['ARM']=='joint':
 tool=3
 codes={dynamic_cp_plan.build_encoder_dynamic_plan.__code__:'encoder',
        dynamic_cp_plan.build_decoder_dynamic_plan.__code__:'decoder',
        dynamic_cp_d4_encoder_capture._validate_capture_context.__code__:'source'}
 assert sys.monitoring.get_tool(tool) is None
 sys.monitoring.use_tool_id(tool,'proxy-first-plans')
 def returned(code,instruction,value):
  role=codes.pop(code)
  sys.monitoring.set_local_events(tool,code,0)
  if role=='source':
   authority,lane,is_source=value
   rank=int(os.environ['RANK']); domain=tuple(authority.domain_ranks)
   assert rank==authority.global_rank and rank in domain and len(domain)==4
   assert type(is_source) is bool and is_source==(rank==domain[0])
   source_info.append((rank,domain,lane,is_source))
  else:
   counts=(collections.Counter(e.group_size for w in value.waves for e in w.executions)
           if role=='encoder' else collections.Counter(a.local_cp_size for m in value.microbatches for a in m.assignments))
   print('PROXY_FIRST_PLAN '+json.dumps(dict(rank=os.environ['RANK'],role=role,
    degrees=dict(counts))),flush=True)
  if not codes:
   if source_info and not source_info[0][3]:
    assert not records,'unexpected non-source sample reader'
    MdpThdMockDataset.__getitem__=original
   sys.monitoring.register_callback(tool,sys.monitoring.events.PY_RETURN,None)
   sys.monitoring.free_tool_id(tool)
 sys.monitoring.register_callback(tool,sys.monitoring.events.PY_RETURN,returned)
 for code in codes:sys.monitoring.set_local_events(tool,code,sys.monitoring.events.PY_RETURN)
try:
 runpy.run_path(str(Path(__file__).resolve().parents[3]/'examples/multimodal_dev/pretrain_multimodal.py'),run_name='__main__')
 if os.environ['ARM']=='joint':assert not codes,'missing native dynamic plan observation'
 if os.environ['ARM']=='joint':
  assert len(source_info)==1,'missing validated source ownership'
  rank,domain,lane,is_source=source_info[0]
  assert len(records)==(64 if is_source else 0),'incorrect source-owned sample window'
  print('PROXY_SOURCE_OWNERSHIP '+json.dumps(dict(rank=rank,domain=domain,lane=lane,
   is_source=is_source,count=len(records))),flush=True)
 else:
  assert len(records)==64,'missing first sample window observation'
finally:
 MdpThdMockDataset.__getitem__=original
 if os.environ['ARM']=='joint' and sys.monitoring.get_tool(tool) is not None:
  for code in codes:sys.monitoring.set_local_events(tool,code,0)
  sys.monitoring.register_callback(tool,sys.monitoring.events.PY_RETURN,None)
  sys.monitoring.free_tool_id(tool)
