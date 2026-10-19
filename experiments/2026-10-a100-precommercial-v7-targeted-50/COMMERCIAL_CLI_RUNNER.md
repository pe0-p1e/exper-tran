# 商业模型 CLI：三图盲测入口

用户指定：GPT通过Codex，Gemini通过Google账号登录的Gemini CLI，Claude通过Antigravity。
这是图片输入/独立会话验证，不是开放问答实验，也不改正在运行的代理研究。

## 唯一入口

```bash
bash experiments/2026-10-a100-precommercial-v7-targeted-50/run_commercial_cli_pilot.sh
```

同一命令续跑，复用已验证的请求。`--preflight`只检查环境与状态；
`--providers gemini`等可由维护者用于接口调试。用户不必手动逐阶段运行。
exit0表示三个target的六次请求全部完成；exit2表示有真实认证、模型配置或接口阻塞，
同时报告已完成的独立结果，缺失结果为N/A。

完整数据/原始回答/请求尝试保存在根盘：
`outputs/a100_precommercial_v7_targeted50/proxy_composition_18h/commercial_cli/`。
统一状态/报告在同级`commercial_cli_campaign/`。
仅compact报告、回答与测量上传GitHub，不上传OAuth凭据、验证码或完整请求体。

## 模型配置与现状

配置：`commercial_cli_targets.json`。模型ID必须来自认证后可用列表，不能使用auto/latest或猜测ID。
Codex固定`gpt-6.1-sol`、CLI0.163.0；Gemini CLI0.64.0；Antigravity1.4.2。
Gemini与Claude的model目前为null，原因是账号授权未完成，尚未核实真实可用ID。
维护者在授权后核实并冻结ID，不从攻击成功率选择模型。
每provider在第一次执行时冻结模型、客户端版本、图片/提示hash和请求顺序；不覆盖不同身份的旧结果。
闭源实际serving snapshot/参数精度不可见时明确记录未知，不宣称已审计闭源BF16。

已有GPT三图六次请求直接复用：clean3/3正确、adv目标命中0/3。
Gemini/Claude尚无有效测量；配置/认证未就绪不是模型预测错误。

## 图片与隔离

- 使用已有固定T03/T05/T09开发图各一张，clean/adv分开，共6请求/target，三个target总计18。
- 校验与冻结source state和PNG的hash对应、224px和最大16/255；不重新生成或筛选图片。
- 每次新进程、独立临时目录、opaque image.png，不传GT、源/目标类别、attack状态或历史回答。
- 原十类提示和精确0–9首行解析；无效回答属于完成但未成功的预测，不能筛掉。
- 访问/传输/图片接口错误不计作有效预测；未收齐三图时主ASR为N/A，分母计划值仍记录3。
- 原始JSONL、stderr、时间、版本、hash、会话ID和错误尝试保存在本地。
- 不允许API-key、Vertex或付费路由自动fallback；不让商业结果回流调参。

## 三种实际传输

1. Codex：已执行的`--image`原生附件；复核原始完成事件、模型catalog图片能力、hash和六个不同会话。
2. Gemini：`@image.png`由客户端本地展开。安装版本的处理器直接实例化ReadManyFilesTool，
   不依赖向模型开放read_file。系统配置限制模型工具为空，禁skills/MCP/额外context/hooks。
   使用仅作用于本次子进程的Node fetch包装器，在出站generateContent请求前核对单张PNG字节hash和model。
   只记录hash/模型/时间/host/path，不记录header、token、OAuth交换或请求正文。
   Gemini标准BeforeModel hooks会过滤非文本媒体，不能拿它的文字日志冒充图片证明。
   如果SDK未走被审计的fetch、请求schema不同或图片被改写，则拒绝计分并保留接口阻塞。
3. Antigravity：官方stream-json stdin仅支持文本块，不支持直接传image block。
   使用新建主agent，仅允许view_file读取本次PNG。核验init model/tools、一轮新会话、成功PNG读取及工具参数。
   任何额外工具、subagent、错误路径、未确认图片读取均拒绝计分。
   该CLI不暴露后端媒体序列化，所以证据边界是成功的本地PNG原生工具读取，serving内部仍不可见。

## 已验证与未验证

CPU/离线测试覆盖：严格解析、模型fallback拒绝、缺失/错误图片拒绝、额外工具拒绝、
会话隔离、无效回答分母保留、出站PNG与model的审计逻辑。
已实际执行统一入口并成功复用Codex六次请求；Gemini/Claude均被正确阻断。
Gemini/Claude适配器尚未经过认证后的真实vision调用，不能把离线测试写成端到端通过。

默认不自动重试失败请求；修复传输接口后可显式`--retry-failed`，逐次保留attempt记录。
已经收到的有效模型预测（包括拒绝和错误类别）不会因结果不好而重试。

## 原始文档

- https://developers.openai.com/codex/cli/reference
- https://geminicli.com/docs/cli/headless/
- https://geminicli.com/docs/reference/configuration/
- https://geminicli.com/docs/hooks/reference/
- https://www.antigravity.google/docs/cli/headless/
- https://www.antigravity.google/docs/subagents/
