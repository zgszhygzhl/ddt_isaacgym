## GitHub Copilot Chat

- Extension: 0.49.0 (prod)
- VS Code: 1.121.0 (f6cfa2ea2403534de03f069bdf160d06451ed282)
- OS: linux 5.15.0-176-generic x64
- Remote Name: ssh-remote
- Extension Kind: Workspace
- GitHub Account: hfdshdy

## Network

User Settings:
```json
  "http.proxy": "http://127.0.0.1:7890",
  "http.systemCertificatesNode": true,
  "github.copilot.advanced.debug.useElectronFetcher": true,
  "github.copilot.advanced.debug.useNodeFetcher": false,
  "github.copilot.advanced.debug.useNodeFetchFetcher": true
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 20.205.243.168 (11 ms)
- DNS ipv6 Lookup: Error (12 ms): getaddrinfo ENOTFOUND api.github.com
- Proxy URL: http://127.0.0.1:7890 (0 ms)
- Proxy Connection: Error (3 ms): connect ECONNREFUSED 127.0.0.1:7890
- Electron fetch: Unavailable
- Node.js https: Error (2 ms): Error: Failed to establish a socket connection to proxies: PROXY 127.0.0.1:7890
	at PacProxyAgent.<anonymous> (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/node_modules/@vscode/proxy-agent/out/agent.js:120:19)
	at Generator.throw (<anonymous>)
	at rejected (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/node_modules/@vscode/proxy-agent/out/agent.js:6:65)
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
- Node.js fetch (configured): Error (11 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14902:13
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
	at async n._fetch (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5484:5278)
	at async n.fetch (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5484:4590)
	at async u (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5516:186)
	at async Sg._executeContributedCommand (file:///root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/out/vs/workbench/api/node/extensionHostProcess.js:502:48807)
  Error: connect ECONNREFUSED 127.0.0.1:7890
  	at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1637:16)

Connecting to https://api.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.112.22 (9 ms)
- DNS ipv6 Lookup: Error (10 ms): getaddrinfo ENOTFOUND api.githubcopilot.com
- Proxy URL: http://127.0.0.1:7890 (0 ms)
- Proxy Connection: Error (1 ms): connect ECONNREFUSED 127.0.0.1:7890
- Electron fetch: Unavailable
- Node.js https: Error (2 ms): Error: Failed to establish a socket connection to proxies: PROXY 127.0.0.1:7890
	at PacProxyAgent.<anonymous> (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/node_modules/@vscode/proxy-agent/out/agent.js:120:19)
	at Generator.throw (<anonymous>)
	at rejected (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/node_modules/@vscode/proxy-agent/out/agent.js:6:65)
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
- Node.js fetch (configured): Error (8 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14902:13
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
	at async n._fetch (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5484:5278)
	at async n.fetch (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5484:4590)
	at async u (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5516:186)
	at async Sg._executeContributedCommand (file:///root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/out/vs/workbench/api/node/extensionHostProcess.js:502:48807)
  Error: connect ECONNREFUSED 127.0.0.1:7890
  	at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1637:16)

Connecting to https://copilot-proxy.githubusercontent.com/_ping:
- DNS ipv4 Lookup: 4.249.131.160 (10 ms)
- DNS ipv6 Lookup: Error (11 ms): getaddrinfo ENOTFOUND copilot-proxy.githubusercontent.com
- Proxy URL: http://127.0.0.1:7890 (1 ms)
- Proxy Connection: Error (1 ms): connect ECONNREFUSED 127.0.0.1:7890
- Electron fetch: Unavailable
- Node.js https: Error (2 ms): Error: Failed to establish a socket connection to proxies: PROXY 127.0.0.1:7890
	at PacProxyAgent.<anonymous> (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/node_modules/@vscode/proxy-agent/out/agent.js:120:19)
	at Generator.throw (<anonymous>)
	at rejected (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/node_modules/@vscode/proxy-agent/out/agent.js:6:65)
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
- Node.js fetch (configured): Error (7 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14902:13
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
	at async n._fetch (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5484:5278)
	at async n.fetch (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5484:4590)
	at async u (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5516:186)
	at async Sg._executeContributedCommand (file:///root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/out/vs/workbench/api/node/extensionHostProcess.js:502:48807)
  Error: connect ECONNREFUSED 127.0.0.1:7890
  	at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1637:16)

Connecting to https://mobile.events.data.microsoft.com: Error (7 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14902:13
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
	at async n._fetch (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5484:5278)
	at async n.fetch (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5484:4590)
	at async u (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5521:135)
	at async Sg._executeContributedCommand (file:///root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/out/vs/workbench/api/node/extensionHostProcess.js:502:48807)
  Error: connect ECONNREFUSED 127.0.0.1:7890
  	at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1637:16)
Connecting to https://dc.services.visualstudio.com: Error (6 ms): TypeError: fetch failed
	at node:internal/deps/undici/undici:14902:13
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
	at async n._fetch (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5484:5278)
	at async n.fetch (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5484:4590)
	at async u (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/extensions/copilot/dist/extension.js:5521:135)
	at async Sg._executeContributedCommand (file:///root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/out/vs/workbench/api/node/extensionHostProcess.js:502:48807)
  Error: connect ECONNREFUSED 127.0.0.1:7890
  	at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1637:16)
Connecting to https://copilot-telemetry.githubusercontent.com/_ping: Error (2 ms): Error: Failed to establish a socket connection to proxies: PROXY 127.0.0.1:7890
	at PacProxyAgent.<anonymous> (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/node_modules/@vscode/proxy-agent/out/agent.js:120:19)
	at Generator.throw (<anonymous>)
	at rejected (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/node_modules/@vscode/proxy-agent/out/agent.js:6:65)
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
Connecting to https://copilot-telemetry.githubusercontent.com/_ping: Error (2 ms): Error: Failed to establish a socket connection to proxies: PROXY 127.0.0.1:7890
	at PacProxyAgent.<anonymous> (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/node_modules/@vscode/proxy-agent/out/agent.js:120:19)
	at Generator.throw (<anonymous>)
	at rejected (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/node_modules/@vscode/proxy-agent/out/agent.js:6:65)
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)
Connecting to https://default.exp-tas.com: Error (2 ms): Error: Failed to establish a socket connection to proxies: PROXY 127.0.0.1:7890
	at PacProxyAgent.<anonymous> (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/node_modules/@vscode/proxy-agent/out/agent.js:120:19)
	at Generator.throw (<anonymous>)
	at rejected (/root/.vscode-server/cli/servers/Stable-f6cfa2ea2403534de03f069bdf160d06451ed282/server/node_modules/@vscode/proxy-agent/out/agent.js:6:65)
	at process.processTicksAndRejections (node:internal/process/task_queues:103:5)

Number of system certificates: 434

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).