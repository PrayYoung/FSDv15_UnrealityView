from pathlib import Path
from tools.remote_runner.cuda_install import patch


def test_host_cuda_filter_precedes_rental_and_excludes_incompatible_offers():
    source=patch((Path.home()/'vast-run/vast_run.py').read_text())
    assert patch(source)==source
    start=source.index('def search_offers(');end=source.index('\ndef choose_offer(',start)
    queries=[]
    def offers(argv):
        queries.append(argv[-1])
        return [dict(gpu_name='RTX 3090',num_gpus=1,host_id=i,cuda_max_good=cuda) for i,cuda in enumerate([12.2,12.8,None])]
    namespace=dict(ToolConfig=object,Any=object,vast_json=offers,InfrastructureError=RuntimeError)
    exec(source[start:end],namespace)
    result=namespace['search_offers'](None,gpu_model='RTX 3090',num_gpus=1,min_reliability=.99,disk_gb=40,min_cuda_version=12.4)
    assert [row['host_id'] for row in result]==[1]
    assert 'cuda_max_good>=12.4' in queries[-1]
    assert len(namespace['search_offers'](None,gpu_model='RTX 3090',num_gpus=1,min_reliability=.99,disk_gb=40))==3
