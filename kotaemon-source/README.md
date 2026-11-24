# Kotaemon engineering knowledge assistant

## 本机多项目工作台

在本文所述的专用 Python 环境准备完成后，从此目录运行：

    & "$env:LOCALAPPDATA\RFQKotaemon\runtime\.venv\Scripts\python.exe" -B .\web_app.py --port 7866

工作台默认监听本机回环地址。新项目默认使用本地检索和原文摘录。资料获准发送给企业模型后，先按下文配置已批准的企业端点，再在该项目的设置中启用“企业模型问答”。工作台使用已有的企业 Azure 适配器。

旧工作台数据库首次打开时，旧版入库的 PDF、Excel 和曾以失败占位文字入库的文件会标记为“待重新解析”。到资料页重新解析这些文件以恢复检索，原件保留在已有项目中。答复只使用候选检索结果中的证据，界面会显示实际证据条数和检查范围。

Makita 和中文 RFQ 本地检索评测可运行 `python -B eval_benchmark.py --data-root "<评测资料上一级目录>" --include-cases`。Makita 的片段命中要求单条候选包含完整参考原文。

Source snapshot prepared on 2026-10-29 from the locally maintained application.
The sibling `kotaemon-pilot` remains the existing local installation and is not
modified by this publication. This snapshot includes its current application
modules, UI assets, tests, startup scripts and selected setup manifests, not just
standalone patches from today's work.

## Current functionality

- Project-scoped document snapshots, manual folder scanning, mixed-format
  ingestion and persistent local indexes.
- PDF, DOCX and XLSX parsing; local Microsoft Office conversion for legacy
  formats; optional local OCR with pinned model hashes.
- Keyword and local-vector retrieval, engineering intent handling, evidence
  selection, cited answers and explicit coverage/parse limitations.
- Background query jobs, progress tracking, conversation history and exact-query
  cache reuse without reclassifying previous AI text as source evidence.
- Citation context and physical PDF page previews, saved answer snapshots,
  new XLSX exports, backups and restoration.
- Updated conversation sidebar, hover titles, responsive layout, query status
  and UI assets from the 2026-10-29 local implementation.

These features do not guarantee exhaustive requirements discovery, complete
complex-table recovery or error-free semantic interpretation. Dates do not
establish approval or contractual precedence.

## Publication boundary

No business documents, original-document snapshots, databases, embeddings,
credentials, company endpoint addresses, local verification logs, or machine
installation status reports are included. Project defaults and LAN test
addresses are replaced with generic examples. The `评审后` directory is only a
local input convention, not a document included in this repository.

The authorized source directory defaults to the sibling `评审后` folder. An
authorized administrator may set `RFQ_AUTHORIZED_SOURCE` to another approved
directory before starting the service; changing it is an authorization decision,
not a convenience setting. Existing projects remain bound to the directory they
were created with and refuse to switch.

The snapshot deliberately differs from the local installation's connection
configuration. Before starting it, an authorized administrator must set
`DUERR_API_APPROVED_ENDPOINT` to the approved HTTPS Azure-style chat endpoint.
It defaults to a non-routable example domain, not a company endpoint. If
`DUERR_API_ENDPOINT` is also provided, it must exactly match that approved value.
The allowlist is fixed when the modules are imported: restart after changes.
Read `DUERR_API_KEY` from the process environment or approved secure storage;
never write its value into a script, tracked configuration or command history.
Optional `DUERR_API_VERSION` and `DUERR_MODEL` retain the adapter's defaults.

Setting an endpoint is not authorization to send project data to it. Obtain the
appropriate company and project-data approvals first. The application does not
fall back to public models, external OCR or hosted embeddings.

## Environment and startup

The scripts expect the dedicated Windows Python environment at
`%LOCALAPPDATA%\RFQKotaemon\runtime\.venv` and a fixed upstream Kotaemon checkout
at `%LOCALAPPDATA%\RFQKotaemon\runtime\kotaemon`.
Upstream source revision: `9ad3e4e49aa35b8acddd235918a5d9753c1cfdf9`
(Kotaemon 0.12.0). Models and runtime data are provisioned separately; this is
not a self-contained installer.

`environment-setup` contains dependency constraints, setup helpers and artifact
hash manifests. Review downloads and dependencies before provisioning. The
Microsoft Office converter requires an approved local Word/Excel installation;
OCR dependencies are listed separately in `requirements-ocr.txt`.

Once the approved runtime, local NLTK resources and model artifacts are present,
start from this directory:

```powershell
.\start-project.ps1 -Project project-demo
```

By default the UI listens only on `http://127.0.0.1:7863`. Local runtime data
resides under `%LOCALAPPDATA%\RFQKotaemon`, outside this repository.
Do not run this snapshot and the original local application against the same
project concurrently. Existing data remains on the original machine and is not
restored simply by cloning the repository.

LAN mode is optional and has no built-in enterprise authentication. It does not
change Windows firewall rules or request administrator elevation. Use it only
with approved network access and document-sharing scope. The
`enterprise-access.conf` file is a reverse-proxy integration template, not an
implemented identity provider or evidence of approved deployment.

Runtime paths can be relocated with absolute `RFQ_RUNTIME_HOME`,
`RFQ_RUNTIME_VENV`, `RFQ_NLTK_DATA`, `RFQ_TIKTOKEN_DATA`, `RFQ_EMBEDDING_DIR`,
`RFQ_OCR_ROOT`, `RFQ_PREVIEW_DATA`, and `RFQ_SOURCE_SCANS` settings; unset values retain the
Windows defaults. The opt-in `RFQ_APP_SERVICE=true` mode binds to the container
interface and requires `RFQ_SINGLE_PROJECT_MODE=true`, an existing absolute
`RFQ_AUTHORIZED_SOURCE` directory, and an HTTPS `RFQ_PUBLIC_ORIGIN` (or
App-Service-provided `WEBSITE_HOSTNAME`). By default it rejects requests without
an Entra `X-MS-CLIENT-PRINCIPAL` and the configured host/origin. This header
check is only a guardrail: App Service Authentication must require sign-in
before traffic reaches the container. Restrict the Entra enterprise
application to the approved user group; this PoC does not implement per-project
user roles.

An explicitly approved alternative, `RFQ_APP_SERVICE_ACCESS_MODE=company-ip`,
permits requests without user sign-in while retaining host/origin checks. This
setting does **not** create or enforce an IP firewall. Before using it, the
deployment script must verify a deny-by-default App Service policy that allows
only the approved company egress IPv4 `/32` and applies the same rule to SCM.
Everyone using that egress can access and modify the data, with no user-level
authorization or audit. A later firewall change can expose the app; the script
does not continuously monitor Azure settings. Use this mode only with explicit
approval for all users sharing the egress and for the selected data.

The app's SQLite databases, original snapshots, indexes, backups, and results
belong under a persistent data path such as `/home/rfq-data`. For Linux custom
containers, enable `WEBSITES_ENABLE_APP_SERVICE_STORAGE=true` and keep the PoC
at one instance with scale-out disabled. The authorized source directory must
already exist (for example, through an IT-approved storage mount);
browser uploads are intentionally disabled. The service lock is an
operating-system advisory lock, and correctness on the selected network
filesystem has not been established. Legacy `.doc`, `.xls`, and `.wps`
conversion remains Windows/Microsoft Office-only.

This startup contract is not a deployment-ready image. A Linux custom-container
build, its App Service storage behavior, and live Entra sign-in have not been
validated here. The built-in Python PoC's single-instance persistence was tested
separately. The embedding manifest identifies a third-party copy of
Xenova's ONNX conversion of BAAI's `bge-small-zh-v1.5`. All six checked-in
manifest artifacts match the public Xenova metadata at revision
`75c43b069aac4d136ba6bc1122f995fedcfd2781` (Git blob IDs for the five
supporting files, SHA-256 for the ONNX weights). The original
[model card](https://huggingface.co/BAAI/bge-small-zh-v1.5) declares MIT, and
its license notice is included with the staged model. The
[conversion card](https://huggingface.co/Xenova/bge-small-zh-v1.5) does not
state a separate license; company approval remains required.

## Azure App Service Python 代码部署

App Service's built-in Linux Python 3.11 runtime is the first deployment option;
Docker and ACR are not required for this route. Azure's Oryx build installs the
Python dependencies from the generated root `requirements.txt`, including
Kotaemon and Ktem from the pinned upstream Git revision. The build therefore
requires Git and access to GitHub and the configured Python package index.
Runtime assets are staged into the ZIP with their manifest checksums; source
documents, credentials, local databases, logs, and the Windows virtual
environment are excluded.

The ZIP includes the existing BGE ONNX model, offline NLTK and tiktoken files,
Kotaemon icons, and the OCR model manifest. Confirm that the company authorizes
uploading these public third-party runtime assets to the test subscription
before deployment. The ONNX conversion's separate license remains unresolved.

Prepare a Linux App Service plan and Web App with the built-in Python 3.11
runtime in the approved test region. Use one worker only and enable HTTPS Only.
The default deployment requires App Service Authentication and Microsoft
Entra sign-in for every request. Restrict enterprise-application assignment to
the approved users or group before publishing the app. The deployment script
does not verify group membership or create an Entra application.

The built-in runtime cannot install arbitrary Linux operating-system packages.
Each deployment must therefore verify Oryx dependency installation,
application startup, PDF/OCR imports and local model initialization. The
existing IP-only PoC was tested; App Service-managed identity headers and live
Entra sign-in remain unverified. If another required system library is absent,
use the separately documented container alternative below.

From `kotaemon-source`, build a source-only package using the already prepared
local RFQ Python runtime and its verified assets:

```powershell
.\environment-setup\build-appservice-code.ps1 `
  -OutputPath "$env:TEMP\rfq-appservice-poc.zip"
```

For Linux x86-64 OCR, this also downloads the official headless OpenCV wheel
specified by `environment-setup\opencv-headless-manifest.json`, verifies its
size and SHA-256, and bundles its license notices. Supply an already downloaded
wheel with `-OpenCvHeadlessWheel "<path>"` for an offline build. The isolated
headless copy is used by the OCR worker because the built-in Python image may
not include `libGL.so.1`; other platforms retain their existing OpenCV setup.

Once IT has provisioned the Web App, enabled and tested Entra authentication,
and confirmed the one-worker plan, upload the package from the signed-in Azure
CLI session:

```powershell
.\environment-setup\deploy-appservice-code.ps1 `
  -ResourceGroup "<resource-group>" `
  -WebAppName "<IT-provided-web-app-name>" `
  -SubscriptionId "<approved-subscription-id>" `
  -PackagePath "$env:TEMP\rfq-appservice-poc.zip"
```

If Azure CLI is not on `PATH`, provide its absolute `az.cmd` path with
`-AzureCli "<path>"`. The deployment command checks the signed-in subscription
against the explicitly supplied approved subscription ID and verifies that the
target is Linux/Python 3.11, HTTPS-only, Entra-authenticated, and on a
single-worker Linux plan. It sets Oryx build mode and the App Service startup
command, then submits the ZIP without waiting for a long Oryx build. Check the
deployment status and app health before treating it as successful or retrying
a timed-out request. It does not create a plan, Web App, Entra app, document
mount, API credential, or business-data copy.

For an explicitly approved no-login deployment, leave the existing access rules
in place and instead pass `-CompanyIpOnly -CompanyEgressCidr "<IPv4>/32"`.
The script refuses to deploy if the web rules allow any other address, if
either endpoint is not deny-by-default, or if SCM does not share the web rules.
It does not change those rules or disable existing authentication. Do not treat
the firewall as identity verification or as a private network.

The deployment script configures application data at `/home/rfq/data` and the
authorized source at `/home/rfq/source`; the source directory must already
exist. For a synthetic smoke test, `-RuntimeRoot /home/rfq-smoke` selects
separate data and source directories under `/home`. The source ZIP contains no company
documents. A first indexed snapshot
can be transferred separately with `export_appservice_project.py` and
`import_appservice_project.py`, which verify file hashes and refuse to replace
an existing `/home/rfq`. Old scan and ingestion receipts are preserved under
`migration-audit`; run a new scan after import. A repeatable update/merge
procedure for existing cloud projects is not implemented. Keep scale-out
disabled because project data uses SQLite and a local advisory lock.

For an initial migration, stop the local project service and run:

```powershell
python .\environment-setup\export_appservice_project.py `
  --project-dir "<local-data>\projects\<project-id>" `
  --source-dir "<authorized-source>" `
  --output "<secure-local-path>\rfq-transfer.zip"
```

The archive contains business documents and indexes: never commit it or put it
under `wwwroot`. Transfer it only through the authenticated, IP-restricted SCM
endpoint. On the Web App's Linux host, run the importer with that archive and
the SHA-256 printed by the exporter:

```bash
python3 import_appservice_project.py --archive /home/<transfer>/rfq-transfer.zip --sha256 <sha256>
```

The importer requires `/home/rfq` not to exist, verifies all archived files and
source snapshots, then installs the source and project together. Delete the
temporary transfer copy only after verification; retain the original source and
local project backup. Legacy Office conversion still requires Windows Office.

Company API settings must be configured separately through an approved secure
mechanism before model-assisted answers can work.

After an Entra deployment, verify `https://<web-app-host>/project-health` while
signed in, then verify an unassigned user is denied. For an IP-only deployment,
verify access from the approved egress and rejection from a different one. Use
only a synthetic fixture to
test scanning, indexing, retrieval, OCR, restart persistence, and backup
restore. The application stores no API keys in the ZIP; configure the approved
company API endpoint and key through Azure's protected app settings or a
Key Vault reference. Do not put keys in shell history or send them in chat.

## Azure App Service container alternative

If the built-in Python runtime fails because a required OS-level library is
unavailable, the Linux container recipe remains available as a fallback. It
uses the same company-chat adapter and BGE embedding model and contains no API
credentials. From this directory, run:

```powershell
.\environment-setup\build-appservice.ps1
```

The script verifies the local BGE, NLTK archives and tiktoken files against
checked-in hashes, stages only the runtime source and required assets in a temporary build
context, and builds `rfq-kotaemon-appservice:local`. It deliberately excludes
business documents, test fixtures, local databases, logs, and credentials.
The image includes local OCR; legacy `.doc`, `.xls`, and `.wps` conversion is
not available on Linux.

Before publishing the image to the approved registry, confirm that the BGE,
NLTK and tiktoken resources are authorized for company Azure use. Configure
the App Service with `WEBSITES_ENABLE_APP_SERVICE_STORAGE=true`,
`WEBSITES_PORT=8000`, and one instance with scale-out disabled. Set
`DUERR_API_APPROVED_ENDPOINT`, `DUERR_API_ENDPOINT`, and `DUERR_API_KEY` using
the approved secure configuration mechanism; `DUERR_API_ENDPOINT` must match
the approved endpoint exactly. Set `DUERR_MODEL` and `DUERR_API_VERSION` only
if the current adapter defaults are not appropriate. Never place secrets in
the image or repository.

Mount or provision the authorized project documents at
`/home/rfq-source` (or set `RFQ_AUTHORIZED_SOURCE` to the approved absolute
path), and set `RFQ_PUBLIC_ORIGIN` if the App Service hostname should not be
used. Configure App Service Authentication with Microsoft Entra ID, require
authentication, and restrict enterprise-application assignment to the
approved users/groups before exposing the service. This PoC uses one project
and does not implement per-user roles.

The image must be built and tested on Linux before deployment. A successful
local image build would still not validate App Service storage, Easy Auth,
model/API connectivity, or the company-authorized document mount.

## GitHub Actions 容器部署指南（备用）

工作流使用 `kotaemon-source` 中的真实应用，默认不会自动部署；必须先验证镜像和目标 Azure
环境。仅有工作流配置不代表应用已经部署。开始前先确认：

- 公司允许该仓库的应用代码由 GitHub 托管的 `ubuntu-latest` runner 构建，并允许从指定来源下载依赖。
  镜像中的 BGE、NLTK 和 tiktoken 资源分别获得企业使用和分发许可；BGE ONNX 文件来自第三方副本，
  其许可尚未确认。未批准前不要启动镜像构建。
- IT 确认测试订阅、Azure 策略、费用、数据分类、访问控制、文件存储和备份符合要求。
  首次用虚构测试文件；是否可放入真实资料，仍以具体存储、人员及数据范围的批准为准。
- GitHub 托管 runner 能连接 ACR；如果 ACR 仅允许私网访问，应先由 IT 提供合规方案。
  当前工作流写死 `ubuntu-latest`，**不能直接用于仅私网可访问的 ACR**。

### 1. Azure 资源和身份（你与 IT）

在测试订阅中按公司规范准备 Linux 容器 Web App、App Service Plan、ACR 和获准的资料存储。
Web App 固定 **1 个实例**、关闭自动扩缩容；本版本只运行一个项目，所有获准用户共享该项目，
不提供项目内按人分权。数据库、原件快照、索引、结果和备份保存在 `/home/rfq-data`。
该网络存储上的 SQLite 和文件锁尚未验证，必须先用虚构数据测试重启、恢复及锁行为；
不要把启用持久化当成备份或多实例安全证明。

请区分三个身份：

1. **访问者登录身份**：为 Web App 启用 App Service Authentication / Microsoft Entra ID，
   要求所有请求登录，未登录请求重定向或拒绝；在对应的企业应用中启用用户分配要求，
   仅分配获准的员工或组。开启 HTTPS Only。先确认无权用户被拒绝。
2. **Web App 托管身份**：授予 ACR 拉取权限，配置 `acrUseManagedIdentityCreds=true`。
   ACR 使用传统 RBAC 时通常为 `AcrPull`；若启用了仓库级 ABAC，使用对应的仓库读取角色。
   按 IT 策略在 Web App 容器设置中预先配置目标 ACR 地址和网络，并确认 ACR 接受 ARM audience token。
   本工作流不修改现有注册表地址或凭据；如需 `DOCKER_REGISTRY_SERVER_URL`，应预先指向
   `https://<ACR名称>.azurecr.io`，不要沿用其他注册表的配置。
   Key Vault 引用如有使用，还需单独授权该身份读取对应密钥。
3. **GitHub 部署身份**：使用单独的 Entra 应用注册或用户分配托管身份建立 OIDC 联合凭据；
   分配 ACR 推送和 Web App 配置更新所需的最小权限。传统 RBAC 的推送角色通常为 `AcrPush`；
   仓库级 ABAC 应使用对应的仓库写入角色，并确保能读取注册表元数据。

部署身份虽然无需读取应用设置，也能通过替换容器间接接触应用数据和密钥，因此必须保护
GitHub 的 `main` 分支、工作流修改权限和部署环境。App Service 管理员、项目资料提供者
和部署身份均应仅限获准人员；工作流检查不代替 IT 安全审查。

在 Web App 的 **环境变量 / 应用设置**中设置：

| 名称 | 值或用途 |
|---|---|
| `WEBSITES_PORT` | `8000` |
| `WEBSITES_ENABLE_APP_SERVICE_STORAGE` | `true` |
| `RFQ_AUTHORIZED_SOURCE` | 已存在的获准资料目录，例如挂载到 `/mnt/rfq-source` |
| `RFQ_PROJECT_ID` | 项目标识，例如 `rfq-poc`；项目创建后不要随意切换资料目录 |
| `RFQ_PUBLIC_ORIGIN` | 可选；仅使用默认域名时可省略，否则填写实际 HTTPS 来源，不带路径 |
| `DUERR_API_APPROVED_ENDPOINT` | 已批准的公司 Azure 对话接口基础地址 |
| `DUERR_API_ENDPOINT` | 可省略；如果填写，必须与上述批准地址完全相同 |
| `DUERR_API_KEY` | 通过公司批准的安全机制配置，优先使用 Key Vault 引用 |
| `DUERR_MODEL` / `DUERR_API_VERSION` | 仅在现有默认值不适用时设置 |

资料目录不是网页上传功能：由 IT 通过批准的存储挂载或其他受控方式提供文件。
挂载前确认目标目录为空、已备份，避免遮盖或删除已有内容；不要把业务文件放进仓库、
镜像或 GitHub Actions artifact。公司 API 从 Azure 出站的网络及 DNS 也需要 IT 确认。
不要在聊天、命令行历史、GitHub secrets、工作流日志或仓库中发送或保存公司 API 密钥。

### 2. 创建受保护的 GitHub 环境（仓库管理员）

在仓库 **Settings → Environments** 检查 `azure-poc`；如果不存在则创建。在 **Deployment branches and tags**
中仅允许 `main`，并在支持时配置 **Required reviewers**。确保 `main` 的推送及工作流变更也受保护。
环境分支限制和 `main` 分支保护都是 OIDC 的安全边界；如果无法落实，先联系 IT，
**不要配置下面的联合凭据或运行部署**。

在 `azure-poc` 环境中配置以下 **Environment secrets**（不要填公司 API Key）：

| 名称 | 内容 |
|---|---|
| `AZURE_CLIENT_ID` | 上述 GitHub 部署身份的 Client ID |
| `AZURE_TENANT_ID` | 企业租户 ID |
| `AZURE_SUBSCRIPTION_ID` | 测试订阅 ID |

在同一环境的 **Environment variables** 中配置：

| 名称 | 内容 |
|---|---|
| `AZURE_RESOURCE_GROUP` | Web App 的资源组名称 |
| `AZURE_WEBAPP_NAME` | Web App 名称，不带域名 |
| `AZURE_ACR_NAME` | ACR 名称，不带 `.azurecr.io` |
| `AZURE_ACR_RESOURCE_GROUP` | ACR 所在资源组名称，可以与 Web App 不同 |

### 3. 配置 GitHub OIDC 联合凭据（Entra 管理员）

在上述**部署身份**中添加 Federated credential：

```text
Issuer:   https://token.actions.githubusercontent.com
Audience: api://AzureADTokenExchange
Subject:  <从当前 GitHub 仓库 OIDC 令牌核实的 subject>
```

联合凭据要求 **subject 精确匹配**；
仓库迁移、重命名或 OIDC 设置变化后应重新核实当前值。该 subject 包含环境而非分支，
因此不能省略第 2 步的环境分支保护。
参考 [GitHub OIDC subject 文档](https://docs.github.com/actions/reference/security/oidc)。

### 4. 先构建验证，再手动部署

经授权后，将相关代码提交、推送并合并到默认分支 `main`。`workflow_dispatch` 工作流文件
必须先存在于默认分支，GitHub 网页才会显示手动运行入口。进入 **Actions → RFQ App Service →
Run workflow**，选择 `main`：

1. 首次只验证容器：勾选 `approve_build`，不要勾选 `deploy_to_azure` 或
   `deployment_prepared`。工作流构建镜像并使用虚构数据检查启动、认证边界、录入、检索和 OCR，
   不推送镜像、不连接公司 API。
2. 构建通过且 Azure、授权用户、存储挂载、单实例和上述应用设置均已准备好后，重新运行；
   勾选 `approve_build`、`deploy_to_azure` 和 `deployment_prepared`。如配置了环境审批，通过审批后，
   工作流以 OIDC 登录、检查部分安全配置、将镜像推送至 ACR，再更新测试 Web App 的镜像。
   每次运行使用包含提交、运行 ID 和重试次数的不同镜像标签；保留与清理策略由 IT 确定。
3. 在 Azure 确认镜像拉取及容器启动成功，用获准和未获准账号分别验证访问控制；
   用虚构文件验证扫描、录入、检索、公司 AI 固定文本连接、重启后的数据及备份恢复。
   仅在安全和数据存储要求满足后使用获准的真实资料。

如需每次向仓库推送相关代码时自动**构建但不部署**，可在公司批准后设置仓库 Actions 变量
`RFQ_GITHUB_HOSTED_BUILD_APPROVED=true`；不设置时，推送事件跳过容器构建。
部署仅可从 `main` 手动触发。

自动检查只覆盖 Easy Auth 的部分配置、HTTPS 和 ACR 托管身份开关，不读取应用设置及密钥；
它不能证明用户组分配、挂载、单实例、存储一致性、镜像拉取或真实登录正确。
容器模拟的身份头不是 Azure Easy Auth 的真实测试。部署成功仅代表镜像配置已更新，
不代表应用已经启动或通过安全验证。

## Offline regression tests

Run with the already provisioned dedicated runtime:

```powershell
& "$env:LOCALAPPDATA\RFQKotaemon\runtime\.venv\Scripts\python.exe" -B -m unittest discover -s tests -v
```

Tests use temporary source fixtures and mocked chat responses. They do not
establish live company API connectivity or authorize access to real documents.
Verification scripts with live modes must be run separately and only within
the authorized data boundary.

GitHub Actions CI runs only the runtime-independent subset (`test_company_probe`,
`test_network_access`, `test_office_convert`, `test_project_registry`,
`test_scan_sources`, `test_synthetic_response` and deployment-related logic).
The additional Linux compatibility job runs only after
`RFQ_GITHUB_HOSTED_BUILD_APPROVED=true`. Run the full suite locally after
changing any other module.
