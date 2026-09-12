"""真实模型候选历史验证；在已配置的模型环境运行，不启动Unity。"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cuda','xpu'),required=True)
    args=parser.parse_args()
    from integrations.motion_service.ardy_generator import ArdyGenerator
    m=ArdyGenerator(model_name='core40',device=args.device,text_device=args.device,checkpoints=os.environ['CHECKPOINTS_DIR'],graph_dir=os.environ.get('YUI_MOTION_GRAPH_DIR'))
    c={'root_xz':[[0.,0.]]*40,'origin_xz':[0.,0.]}
    a,s=m.generate_candidate('A person stands calmly with relaxed arms.',c)
    h=m.history_at(s,4);before=h.clone()
    b,t=m.generate_candidate('A person waves their right hand gently while standing.',c,h)
    assert m.torch.equal(h,before)
    assert t.shape[1]==44 and m.history_at(t,40).shape[1]==20
    report=dict(load_s=m.load_s,warmup_s=m.warmup_s,execution_warmup_s=m.execution_warmup_s,history_frames=m.history_limit,frames=len(b['root_positions']),history_unchanged=True,last_generation_s=m.last_generation_s,text_cache_size=len(m.text_cache),graphs=m.graph.captures if m.graph else 0,fallbacks=m.graph.fallbacks if m.graph else 0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
