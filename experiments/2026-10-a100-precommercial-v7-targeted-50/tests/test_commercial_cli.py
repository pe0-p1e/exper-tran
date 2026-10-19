"""Prevent false successes caused by missing images, fallback, context reuse or errors."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from commercial_cli_adapters import InterfaceBlocker, parse_label, parse_gemini, parse_claude
from run_commercial_cli import validate_record


@pytest.mark.parametrize('raw,label',[('0',1),('9\nexplanation',10),('\n4\n',5),('4.',None),('10',None),('The answer is 4',None),('',None)])
def test_exact_original_parser(raw,label):
    assert parse_label(raw)==label


def gemini_events():
    return [dict(type='init',model='gemini-test',session_id='fresh'),
            dict(type='message',role='assistant',content='4'),dict(type='result',status='success')]


def test_gemini_requires_real_image_audit_and_exact_model():
    events=gemini_events()
    with pytest.raises(InterfaceBlocker):parse_gemini(events,'gemini-test',[])
    with pytest.raises(InterfaceBlocker):parse_gemini(events,'other-model',[dict(status='verified',request_model='gemini-test')])
    with pytest.raises(InterfaceBlocker):parse_gemini(events+[dict(type='tool_use')],'gemini-test',[dict(status='verified',request_model='gemini-test')])
    answer,session,evidence=parse_gemini(events,'gemini-test',[dict(status='verified',request_model='gemini-test')])
    assert answer=='4' and session=='fresh'


def claude_events(image):
    return [dict(event='init',init=dict(model='claude-test',tools=['view_file'])),
            dict(event='step_update',step_update=dict(step_type='tool',state='DONE',tool_name='view_file',
                  tool_info=dict(parameters={'AbsolutePath':str(image)},output='Viewed PNG image'))),
            dict(event='result',result=dict(status='SUCCESS',num_turns=1,response='4',conversation_id='fresh'))]


def test_claude_requires_successful_only_image_read(tmp_path):
    image=tmp_path/'image.png';events=claude_events(image)
    assert parse_claude(events,'claude-test',image)[0]=='4'
    with pytest.raises(InterfaceBlocker):parse_claude([events[0],events[-1]],'claude-test',image)
    with pytest.raises(InterfaceBlocker):parse_claude(events,'claude-test',tmp_path/'other.png')
    events[0]['init']['tools'].append('run_command')
    with pytest.raises(InterfaceBlocker):parse_claude(events,'claude-test',image)


def test_cached_raw_response_cannot_change_parser_or_identity():
    expected={'model':'fixed'}
    validate_record(dict(identity=expected,raw_output='4.',parsed_label=None),expected)
    with pytest.raises(InterfaceBlocker):validate_record(dict(identity=expected,raw_output='4.',parsed_label=5),expected)
    with pytest.raises(InterfaceBlocker):validate_record(dict(identity={'model':'other'},raw_output='4',parsed_label=5),expected)


def test_invalid_replies_stay_in_completed_denominator_and_sessions_must_be_fresh(monkeypatch,tmp_path):
    import run_commercial_cli as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'CONTROL',tmp_path/'control')
    config=dict(providers={'gemini':{'model':'gemini-test'}})
    cases=[]
    for i in range(3):
        requests={'clean':f'c{i}','adversarial':f'a{i}'}
        cases.append(dict(transition=f'T{i}',image_id=str(i),source_label=1,target_label=2,requests=requests))
        for condition,rid in requests.items():
            p=tmp_path/'commercial_cli/gemini/gemini-test'/rid/'result.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(parsed_label=1 if condition=='clean' else None,session_id=rid)))
    rows=runner.aggregate(config,dict(cases=cases),{})
    assert rows[0]['status']=='COMPLETE' and rows[0]['denominator']==3 and rows[0]['hits']==0
    p=tmp_path/'commercial_cli/gemini/gemini-test/a2/result.json'
    p.write_text(json.dumps(dict(parsed_label=None,session_id='c0')))
    with pytest.raises(InterfaceBlocker):runner.aggregate(config,dict(cases=cases),{})


def test_gemini_network_audit_rejects_changed_png_and_model():
    module=(Path(__file__).resolve().parents[1]/'src/gemini_image_transport_audit.mjs').as_uri()
    script=f'''import {{verifyPayload}} from {json.dumps(module)};
import {{createHash}} from 'node:crypto';
const data=Buffer.from('test-image').toString('base64');
const h=createHash('sha256').update(Buffer.from('test-image')).digest('hex');
const payload={{model:'fixed',request:{{contents:[{{parts:[{{inlineData:{{mimeType:'image/png',data}}}}]}}]}}}};
verifyPayload(payload,h,'fixed');
let blocked=0;
for (const [hash,model] of [['wrong','fixed'],[h,'fallback']]) {{
 try {{verifyPayload(payload,hash,model);}} catch {{blocked++;}}
}}
if(blocked!==2)process.exit(1);
'''
    subprocess.run(['node','--input-type=module','-e',script],check=True)
