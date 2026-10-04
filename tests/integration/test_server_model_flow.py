import io
import textwrap
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('fastapi')
pytest.importorskip('httpx')
pytest.importorskip('scipy')
pytest.importorskip('sklearn')
pytest.importorskip('pyarrow')

from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from modellab.server import ServerSettings, create_app

COLORS = {'RED': (200, 50, 40), 'GREEN': (40, 200, 50), 'BLUE': (40, 50, 200)}
NAMES = list(COLORS)
CONFIG = {'layers': [
    {'type': 'AdaptiveAvgPool2d', 'args': {'output_size': 1}}, {'type': 'Flatten', 'args': {}},
    {'type': 'Linear', 'args': {'in_features': 3, 'out_features': 3}},
]}
FACTORY = textwrap.dedent('''
    from torch import nn

    def build():
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))
''')


def identity():
    module = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 3))
    with torch.no_grad():
        module[2].weight.copy_(torch.eye(3) * 10)
        module[2].bias.zero_()
    return module


def checkpoint(tmp_path, name='ck.pt', wrapped=True, obj=None):
    module = identity()
    payload = obj if obj is not None else (
        {'model_state_dict': module.state_dict(), 'class_names': NAMES, 'image_size': 8} if wrapped else module.state_dict()
    )
    path = tmp_path / name
    torch.save(payload, path)
    return path


def dataset_zip():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for cls, color in COLORS.items():
            for i in range(6):
                image = io.BytesIO()
                Image.new('RGB', (8, 8), color).save(image, format='PNG')
                archive.writestr(f'{cls}/{i:03d}.png', image.getvalue())
    return buffer.getvalue()


def stage(client, path, artifact_id, **extra):
    files = {'artifact': (path.name, path.read_bytes(), 'application/octet-stream')}
    files.update(extra.pop('files', {}))
    return client.post('/model-artifacts', data={'artifact_id': artifact_id, **extra}, files=files)


def evaluate(client, model_id, evaluation_id):
    response = client.post('/evaluations', json={'model_id': model_id, 'dataset_id': 'toy', 'device': 'cpu', 'evaluation_id': evaluation_id},
                           params={'wait': True, 'timeout': 600})
    assert response.status_code == 200, response.text
    job = response.json()
    assert job['status'] == 'completed', job
    return job['result']


def test_model_wizard_end_to_end(tmp_path):
    ws = tmp_path / 'ws'
    with TestClient(create_app(ServerSettings(workspace=ws))) as client:
        assert client.post('/datasets', data={'dataset_id': 'toy'}, files={'archive': ('d.zip', dataset_zip(), 'application/zip')}).status_code == 201
        archs = {a['name']: a for a in client.get('/architectures').json()}
        assert {'sequential', 'torchvision'} <= set(archs) and archs['sequential']['config_schema']['required'] == ['layers']

        r = stage(client, checkpoint(tmp_path), 'ck1')
        assert r.status_code == 201, r.text
        staged = r.json()
        assert 'path' not in staged['artifact'] and staged['inspection']['format'] == 'torch_archive'
        assert staged['inspection']['state_dicts'][0]['key_path'] == 'model_state_dict'
        assert {h['field'] for h in staged['inspection']['hints']} >= {'class_names', 'input_size'}
        assert stage(client, checkpoint(tmp_path), 'ck1').status_code == 409
        assert client.post('/model-artifacts', data={}).status_code == 422
        assert [a['artifact_id'] for a in client.get('/model-artifacts').json()] == ['ck1']

        step1 = client.post('/models/resolve', json={'artifact_id': 'ck1', 'probe_architectures': False}).json()
        assert step1['status'] == 'needs_input' and step1['missing'][0]['field'] == 'architecture'
        assert 'sequential' in step1['missing'][0]['options'] and step1['resolved']['num_classes'] == 3
        assert any(i['field'] == 'class_names' and i['applied'] for i in step1['inferred'])
        step2 = client.post('/models/resolve', json={'artifact_id': 'ck1', 'architecture': 'sequential'}).json()
        assert step2['status'] == 'needs_input' and step2['missing'][0]['field'] == 'architecture_config.layers'
        body = {'artifact_id': 'ck1', 'architecture': 'sequential', 'architecture_config': CONFIG}
        step3 = client.post('/models/resolve', json=body).json()
        assert step3['status'] == 'ready' and step3['route'] == 'registered_architecture' and step3['spec']['architecture'] == 'sequential'
        assert [c['check'] for c in step3['validation']['checks']] == ['model_loaded', 'forward_pass']
        assert client.get('/models').json() == []

        broken = {**body, 'architecture_config': {'layers': CONFIG['layers'][:-1] + [{'type': 'Linear', 'args': {'in_features': 3, 'out_features': 4}}]}}
        refused = client.post('/models/from-artifact', json={**broken, 'model_id': 'nope'})
        assert refused.status_code == 422 and refused.json()['detail']['resolution']['status'] == 'invalid'
        assert client.get('/models/nope').status_code == 404

        made = client.post('/models/from-artifact', json={**body, 'model_id': 'wizard'})
        assert made.status_code == 201, made.text
        record = made.json()
        assert record['source'] == 'resolved' and record['spec']['architecture'] == 'sequential' and record['resolution']['route'] == 'registered_architecture'
        assert str(ws / 'models' / 'wizard') in record['spec']['path'] and Path(record['spec']['path']).is_file()
        assert client.post('/models/from-artifact', json={**body, 'model_id': 'wizard'}).status_code == 409
        assert evaluate(client, 'wizard', 'e_wizard')['accuracy'] == 1.0
        meta = client.get('/evaluations/e_wizard').json()['metadata']
        assert meta['model']['architecture'] == 'sequential' and meta['model_id'] == 'wizard'

        spec = {'source': 'state_dict', 'num_classes': 3, 'path': str(checkpoint(tmp_path, 'local.pt')), 'state_dict_key': 'model_state_dict',
                'architecture': 'sequential', 'architecture_config': CONFIG, 'class_names': NAMES}
        legacy = client.post('/models/register', json={'model_id': 'legacy', 'spec': spec})
        assert legacy.status_code == 201 and legacy.json()['check']['loaded'] is True
        assert evaluate(client, 'legacy', 'e_legacy')['accuracy'] == 1.0

        full = tmp_path / 'full.pt'
        torch.save(identity(), full)
        assert stage(client, full, 'pk').json()['inspection']['format'] == 'pickle_object'
        ask = {'artifact_id': 'pk', 'class_names': NAMES, 'input_size': [8, 8]}
        needs = client.post('/models/resolve', json=ask).json()
        assert needs['status'] == 'needs_input' and needs['missing'][0]['field'] == 'trust_executable'
        ok = client.post('/models/resolve', json={**ask, 'trust_executable': True}).json()
        assert ok['status'] == 'ready' and ok['spec']['source'] == 'module'
        assert client.post('/models/from-artifact', json={**ask, 'trust_executable': True, 'model_id': 'pickled'}).status_code == 201
        assert evaluate(client, 'pickled', 'e_pickled')['accuracy'] == 1.0

        fac = {'factory_file': ('factory.py', FACTORY.encode(), 'text/x-python')}
        assert stage(client, checkpoint(tmp_path, 'plain.pt', wrapped=False), 'fk', files=fac).status_code == 201
        fask = {'artifact_id': 'fk', 'factory': 'build', 'class_names': NAMES, 'input_size': [8, 8]}
        assert client.post('/models/resolve', json=fask).json()['missing'][0]['field'] == 'trust_executable'
        ready = client.post('/models/resolve', json={**fask, 'trust_executable': True}).json()
        assert ready['status'] == 'ready' and ready['route'] == 'custom_factory'
        made = client.post('/models/from-artifact', json={**fask, 'trust_executable': True, 'model_id': 'custom'}).json()
        factory_path = Path(made['spec']['factory'].split(':')[0])
        assert factory_path.is_file() and str(ws / 'models' / 'custom') in str(factory_path)
        assert evaluate(client, 'custom', 'e_custom')['accuracy'] == 1.0

        local = client.post('/model-artifacts', data={'artifact_id': 'loc', 'local_path': str(checkpoint(tmp_path, 'l2.pt'))})
        assert local.status_code == 201 and local.json()['artifact']['kind'] == 'local'
        assert client.post('/models/from-artifact', json={**body, 'artifact_id': 'loc', 'model_id': 'from-local'}).status_code == 201
        assert client.delete('/model-artifacts/loc').status_code == 200 and client.get('/model-artifacts/loc').status_code == 404
        assert client.post('/models/resolve', json={'artifact_id': 'missing'}).status_code == 404
        assert client.post('/models/resolve', json={'nope': 1}).status_code == 422
        onnx = tmp_path / 'm.onnx'
        onnx.write_bytes(b'\x08\x07onnx')
        res = client.post('/models/resolve', json={'artifact_id': stage(client, onnx, 'ox').json()['artifact_id']}).json()
        assert res['status'] == 'invalid' and 'ONNX' in res['errors'][0]


def test_server_policy_flags(tmp_path):
    pk = tmp_path / 'full.pt'
    torch.save(identity(), pk)
    settings = ServerSettings(workspace=tmp_path / 'ws', allow_pickle=False, allow_custom_code=False, allow_local_paths=False)
    with TestClient(create_app(settings)) as client:
        assert stage(client, pk, 'pk').status_code == 201
        res = client.post('/models/resolve', json={'artifact_id': 'pk', 'class_names': NAMES, 'trust_executable': True}).json()
        assert res['status'] == 'invalid' and 'disabled' in res['errors'][0]
        assert client.post('/model-artifacts', data={'local_path': str(pk)}).status_code == 403
        files = {'factory_file': ('factory.py', FACTORY.encode(), 'text/x-python')}
        assert stage(client, checkpoint(tmp_path, 'w.pt', wrapped=False), 'fk', files=files).status_code == 201
        res = client.post('/models/resolve', json={'artifact_id': 'fk', 'factory': 'build', 'class_names': NAMES, 'trust_executable': True}).json()
        assert res['status'] == 'invalid' and 'disabled' in res['errors'][0]
