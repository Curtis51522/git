# Bakery AI 最终离线安装说明

## 用户入口

完整解压对应平台的 ZIP 后，直接双击安装器，不需要输入命令行参数：

- Windows x64：`Install Bakery AI.exe`
- Apple Silicon macOS：`BakeryAI Installer.app`

安装器会自动寻找相邻的 `payload` 文件夹、识别平台，并提供应用、数据和备份目录的默认值。用户可以在安装窗口选择安装、修复、升级或卸载。最终启动器必须位于 `payload/launcher/Bakery AI.exe` 或 `payload/launcher/Bakery AI.app`。

## 离线前置依赖

在全新 Windows 电脑上，安装器会在需要时校验并安装离线提供的 WSL、Docker Desktop 和 Microsoft Edge。WSL 需要重启时，安装器会先校验续装状态和 release manifest 的哈希，再通过一次性的 `--resume` 继续安装。在 Apple Silicon macOS 上，安装器会校验并安装 Docker DMG 与 Edge PKG，启动 Docker Desktop，并等待 Docker Engine 就绪后再载入镜像。如果 Docker 无法自动启动，错误信息会明确要求用户启动 Docker Desktop 后重新运行同一个安装器。

## 数据库与卸载

ZIP 中的数据库初始化文件与 Compose 挂载路径完全一致：

```text
payload/deployment/database/init/001-final-snapshot.sql
payload/deployment/database/init/001-final-snapshot.sql.sha256.json
payload/deployment/database/init/999-deployment-ready.sql
```

标准卸载只删除界面中列出的应用和启动器文件，保留营业数据、配置、备份和 MySQL Docker volume。完全删除会显示每个明确文件和受管理的 volume，并要求第二次输入确认。安装器不会跟随符号链接，也不会递归删除目录。

