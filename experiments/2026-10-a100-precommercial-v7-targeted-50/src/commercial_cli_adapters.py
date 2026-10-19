"""CLI-only adapters and strict evidence parsing. No commercial SDK/API fallback."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

from proxy_study_protocol import EXP, sha, atomic_json


class InterfaceBlocker(RuntimeError):
    pass


def parse_label(raw):
    line=next((x.strip() for x in raw.splitlines() if x.strip()),'')
    return int(line)+1 if line in tuple('0123456789') else None


def json_events(raw):
    result=[]
    for line in raw.splitlines():
        if line.strip():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise InterfaceBlocker('Non-JSON output in structured client stream') from exc
    return result


def version(provider):
    exe={'codex':'codex','gemini':'gemini','claude':'agy'}[provider]
    try:
        return subprocess.check_output([exe,'--version'],text=True,stderr=subprocess.PIPE,timeout=20).strip()
    except (FileNotFoundError,subprocess.SubprocessError) as exc:
        raise InterfaceBlocker('Client unavailable: '+exe) from exc


def preflight(provider,spec):
    actual=version(provider)
    if actual!=spec['client_version']:
        raise InterfaceBlocker('Installed client version differs from frozen config: '+actual)
    model=spec.get('model')
    if not model or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{1,120}',model):
        raise InterfaceBlocker('Exact authenticated model ID is not configured; aliases/default/auto are forbidden')
    if model.lower() in ['auto','default','latest','pro','flash']:
        raise InterfaceBlocker('Unpinned model alias')
    if provider=='codex':
        auth=subprocess.run(['codex','login','status'],capture_output=True,text=True,timeout=20)
        if 'Logged in using ChatGPT' not in auth.stdout+auth.stderr:
            raise InterfaceBlocker('Existing ChatGPT CLI login required; API fallback disabled')
        p=Path.home()/'.codex/models_cache.json'
        entries=json.loads(p.read_text()).get('models',[]) if p.exists() else []
        if not any(x.get('slug')==model and 'image' in x.get('input_modalities',[]) for x in entries):
            raise InterfaceBlocker('Model is not in the image-capable Codex catalog')
    elif provider=='gemini':
        p=Path.home()/'.gemini/oauth_creds.json'
        if not p.is_file():
            raise InterfaceBlocker('Google OAuth login has not completed')
        settings=Path.home()/'.gemini/settings.json'
        d=json.loads(settings.read_text()) if settings.exists() else {}
        if d.get('security',{}).get('auth',{}).get('selectedType')!='oauth-personal':
            raise InterfaceBlocker('Gemini must use the selected Google OAuth route')
        if d.get('hooks') or d.get('mcpServers'):
            raise InterfaceBlocker('Unrelated global hooks/MCP present; isolate before the pilot')
    else:
        catalog=subprocess.run(['agy','models'],capture_output=True,text=True,timeout=30)
        if catalog.returncode:
            raise InterfaceBlocker('Antigravity login/catalog unavailable')
        clean=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',catalog.stdout)
        if not re.search(r'(?<![A-Za-z0-9_.-])'+re.escape(model)+r'(?![A-Za-z0-9_.-])',clean):
            raise InterfaceBlocker('Exact Claude model ID not found in authenticated catalog')
        if 'claude' not in model.lower():
            raise InterfaceBlocker('Claude route must actually select a Claude model')
        settings=Path.home()/'.gemini/antigravity-cli/settings.json'
        d=json.loads(settings.read_text()) if settings.exists() else {}
        if any(d.get(k) for k in ['hooks','mcpServers','plugins','skills']):
            raise InterfaceBlocker('Unrelated Antigravity customization requires isolation first')
    return dict(status='READY',client=actual,requested_model=model,
                serving_snapshot='not exposed by CLI; do not claim a verified backend snapshot')


def build(provider,spec,tmp,image,requestdir):
    """Return argv/stdin/env; argv never passes a label or private manifest path."""
    model=spec['model'];env=dict(os.environ)
    if provider=='gemini':
        # Force the user-selected OAuth route, not an inherited paid API environment.
        for key in ['GEMINI_API_KEY','GOOGLE_API_KEY','GOOGLE_GENAI_USE_VERTEXAI','GOOGLE_APPLICATION_CREDENTIALS']:
            env.pop(key,None)
        settings=tmp/'settings.json'
        atomic_json(settings,dict(security={'auth':{'selectedType':'oauth-personal'}},
            context={'fileName':[],'includeDirectoryTree':False,'memoryBoundaryMarkers':[]},
            skills={'enabled':False},hooksConfig={'enabled':False},
            tools={'core':[]},mcp={'allowed':[]},useWriteTodos=False,
            advanced={'ignoreLocalEnv':True},general={'enableAutoUpdate':False}))
        env['GEMINI_CLI_SYSTEM_SETTINGS_PATH']=str(settings)
        env['CLI_IMAGE_AUDIT_FILE']=str(requestdir/'image_transport.jsonl')
        env['CLI_EXPECTED_IMAGE_SHA256']=sha(image);env['CLI_EXPECTED_MODEL']=model
        # Per-child import only. A missing interception makes the observation unverified.
        env['NODE_OPTIONS']='--import='+str(EXP/'src/gemini_image_transport_audit.mjs')
        return ['gemini','--model',model,'--output-format','stream-json','--session-id',str(uuid.uuid4()),
                '--extensions','none','--prompt','@image.png\n'],env
    if provider=='claude':
        agent=tmp/'.agents/agents/image-check.md';agent.parent.mkdir(parents=True,exist_ok=True)
        agent.write_text('''---
name: image-check
description: Read one supplied image and answer its fixed classification question.
tools: [view_file]
mainAgent: true
subagent: false
commandExecutionPolicy: off
mcpServers: []
skills: []
plugins: []
---
Read only the supplied image.png using view_file, then answer the user's question.
Use no other files, tools, agents, memories, search or previous conversations.
''')
        return ['agy','--model',model,'--agent','image-check','--effort','low',
                '--disable-slash-commands','--mode','plan','--output-format','stream-json',
                '--print-timeout','180s','--print'],env
    raise ValueError('Codex uses the existing, already executed adapter')


def parse_gemini(events,model,transport):
    initial=[e for e in events if e.get('type')=='init']
    terminal=[e for e in events if e.get('type')=='result']
    if len(initial)!=1 or initial[0].get('model')!=model:
        raise InterfaceBlocker('Missing or different Gemini model identity')
    if len(terminal)!=1 or terminal[0].get('status')!='success':
        raise InterfaceBlocker('No successful terminal Gemini event')
    if any(e.get('type') in ['tool_use','tool_result','error'] for e in events):
        raise InterfaceBlocker('Unexpected tool use or client error')
    if not transport or any(e.get('status')!='verified' or e.get('request_model')!=model for e in transport):
        raise InterfaceBlocker('Outgoing Gemini image bytes/model could not be verified')
    texts=[e.get('content','') for e in events if e.get('type')=='message' and e.get('role')=='assistant']
    return ''.join(texts),initial[0].get('session_id'),dict(image_requests=len(transport),image_transport='outgoing request PNG SHA256 verified')


def parse_claude(events,model,image):
    initial=[e.get('init',{}) for e in events if e.get('event')=='init']
    result=[e.get('result',{}) for e in events if e.get('event')=='result']
    if len(initial)!=1 or initial[0].get('model')!=model:
        raise InterfaceBlocker('Missing or different Antigravity model identity')
    tools=set(initial[0].get('tools',[]))
    if tools!={'view_file'}:
        raise InterfaceBlocker('Image-only agent tool restriction was not enforced')
    if len(result)!=1 or result[0].get('status')!='SUCCESS' or result[0].get('num_turns')!=1:
        raise InterfaceBlocker('No fresh successful one-turn Claude session')
    steps=[e['step_update'] for e in events if e.get('event')=='step_update']
    reads=[]
    for step in steps:
        if step.get('subagent_info'):
            raise InterfaceBlocker('Unexpected subagent context')
        if step.get('step_type')!='tool':continue
        info=step.get('tool_info',{})
        if step.get('tool_name')!='view_file':
            raise InterfaceBlocker('Unexpected tool in Claude pilot')
        if step.get('state')!='DONE':continue
        supplied=info.get('parameters',{}).get('AbsolutePath','')
        if Path(supplied).resolve()!=image.resolve() or info.get('error'):
            raise InterfaceBlocker('Image tool read failed or accessed another path')
        output=json.dumps(info.get('output','')).lower()
        if not re.search(r'image|png|width|height',output) or re.search(r'unsupported|cannot read|unable to|failed|no inline data',output):
            raise InterfaceBlocker('PNG tool result does not confirm successful image access')
        reads.append(info)
    if not reads:
        raise InterfaceBlocker('No successful PNG view_file in trace; image input unverified')
    return result[0].get('response',''),result[0].get('conversation_id'),dict(image_reads=len(reads),
                image_transport='hash-verified local PNG with successful native view_file; backend media serialization not exposed')
