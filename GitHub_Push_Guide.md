# 将本地项目推送到GitHub仓库的详细指南

## 概述
本指南将帮助您将 `/data2/yihong/ZS-NTTA-master` 文件夹推送到您的GitHub仓库 `https://github.com/Aboriginer/ZS-NTTA-Extension`。

## 前置条件

### 1. 确保Git已安装
在终端中运行以下命令检查Git是否已安装：
```bash
git --version
```

### 2. 配置Git用户信息
如果这是您第一次使用Git，需要配置用户信息：
```bash
# 全局配置（推荐）
git config --global user.name "smlfy"
git config --global user.email "fjsmlfy@gmail.com"

# 或仅为当前仓库配置
git config user.name "smlfy"
git config user.email "your-email@example.com"
```

### 3. GitHub身份验证设置
由于GitHub在2021年8月13日后不再支持密码认证，您需要设置以下任一方式：

#### 方法A：个人访问令牌（Personal Access Token）
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 选择所需权限（至少需要 `repo` 权限）
4. 生成并保存令牌（请妥善保管，只显示一次）

#### 方法B：SSH密钥（推荐）
1. 生成SSH密钥：
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your-email@example.com"
   ```
2. 添加到SSH代理：
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_rsa
   ```
3. 复制公钥内容：
   ```bash
   cat ~/.ssh/id_rsa.pub
   ```
4. 访问 https://github.com/settings/ssh 添加新的SSH密钥

## 详细推送步骤

### 步骤1：导航到项目目录
```bash
cd /data2/yihong/ZS-NTTA-master
```

### 步骤2：初始化Git仓库
```bash
   git init
```

### 步骤3：添加所有文件到暂存区
```bash
git add .
```

### 步骤4：检查文件状态（可选）
```bash
git status
```

### 步骤5：创建初始提交
您可以选择以下任一方式创建提交信息：

#### 方法A：简单单行提交（推荐）
```bash
git commit -m "Initial commit: ZS-NTTA with image similarity extension"
```

#### 方法B：多行详细提交
```bash
git commit -m "Initial commit: ZS-NTTA with image similarity extension

🤖 Generated with Claude Code(https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

#### 方法C：使用编辑器
```bash
git commit
```
然后在打开的编辑器中输入提交信息，保存并关闭。

### 步骤6：添加远程仓库
根据您选择的认证方式，使用以下命令之一：

#### 使用HTTPS（需要个人访问令牌）：
```bash
git remote add origin https://github.com/Aboriginer/ZS-NTTA-Extension.git
```

#### 使用SSH（推荐）：
```bash
git remote add origin git@github.com:Aboriginer/ZS-NTTA-Extension.git
```

### 步骤7：验证远程仓库URL（可选）
```bash
git remote -v
```

### 步骤8：推送到GitHub
```bash
git push -u origin main
```

如果您的默认分支是 `master` 而不是 `main`，请使用：
```bash
git push -u origin master
```

## 可能遇到的问题及解决方案

### 问题1：分支名称不匹配错误
**错误信息**：`error: src refspec main does not match any`

**原因**：本地分支名称与您尝试推送的分支名称不匹配。

**解决方案**：
```bash
# 检查当前分支名
git branch

# 如果显示 * master，使用以下命令推送
git push -u origin master

# 或者将分支重命名为 main
git branch -M main
git push -u origin main
```

### 问题2：远程仓库不为空错误
**错误信息**：`Updates were rejected because the remote contains work that you do not have locally`

**原因**：远程仓库已有内容（如README.md），需要先合并。

**解决方案**：
```bash
# 拉取远程内容并允许不相关历史合并
git pull origin master --allow-unrelated-histories

# 如果出现合并冲突（如README.md冲突）
git status  # 查看冲突文件

# 解决冲突的方法：
# 方法A：保留本地版本
git checkout --ours README.md

# 方法B：保留远程版本  
git checkout --theirs README.md

# 方法C：手动编辑文件解决冲突
nano README.md  # 删除冲突标记，保留需要的内容

# 完成合并
git add .
git commit -m "Merge remote and local repositories"
git push -u origin master
```

**强制推送方案**（会覆盖远程内容）：
```bash
git push -u origin master --force
```

### 问题3：网络连接问题
**错误信息**：`Failed to connect to github.com port 443: Connection timed out`

**原因**：在中国大陆访问GitHub经常遇到网络问题。

**解决方案**：

#### 方法A：使用VPN（推荐）
开启VPN后重新尝试推送。

#### 方法B：配置Git代理
```bash
# HTTP/HTTPS代理
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy https://127.0.0.1:7890

# SOCKS5代理
git config --global http.proxy socks5://127.0.0.1:7890
git config --global https.proxy socks5://127.0.0.1:7890

# 推送完成后取消代理（可选）
git config --global --unset http.proxy
git config --global --unset https.proxy
```

#### 方法C：使用SSH方式
```bash
# 修改远程URL为SSH
git remote set-url origin git@github.com:Aboriginer/ZS-NTTA-Extension.git
git push -u origin master
```

#### 方法D：修改hosts文件
```bash
sudo vim /etc/hosts
```
添加：
```
140.82.113.3 github.com
140.82.114.9 codeload.github.com
```

### 问题4：认证失败
- 如果使用HTTPS，确保使用个人访问令牌而不是密码
- 如果使用SSH，确保SSH密钥已正确添加到GitHub账户

### 问题5：文件太大
如果有大文件（>100MB），考虑使用Git LFS：
```bash
git lfs install
git lfs track "*.大文件扩展名"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

## 后续操作

### 日常提交流程
```bash
# 1. 修改文件后，添加到暂存区
git add .

# 2. 提交更改
git commit -m "描述您的更改"

# 3. 推送到远程仓库
git push origin main
```

### 创建.gitignore文件（推荐）
创建 `.gitignore` 文件忽略不需要版本控制的文件：
```bash
# Python
__pycache__/
*.py[cod]
*.so
.Python
env/
venv/
ENV/

# 数据文件
*.pkl
*.pth
data/
results/
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统文件
.DS_Store
Thumbs.db
```

## 验证推送成功
推送完成后，访问您的仓库地址确认文件已上传：
https://github.com/Aboriginer/ZS-NTTA-Extension

## 相关链接
- GitHub官方文档：https://docs.github.com/en/get-started/importing-your-projects-to-github/importing-source-code-to-github/adding-locally-hosted-code-to-github
- Git官方文档：https://git-scm.com/doc
- GitHub SSH设置：https://docs.github.com/en/authentication/connecting-to-github-with-ssh
- GitHub Personal Access Token：https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

## 完整推送流程总结

基于实际遇到的问题，推荐的完整推送流程：

```bash
# 1. 导航到项目目录
cd /data2/yihong/ZS-NTTA-master

# 2. 初始化Git仓库
git init

# 3. 配置用户信息
git config user.name "smlfy"
git config user.email "fjsmlfy@gmail.com"

# 4. 添加文件到暂存区
git add .

# 5. 创建初始提交
git commit -m "Initial commit: ZS-NTTA with image similarity extension"

# 6. 添加远程仓库
git remote add origin https://github.com/Aboriginer/ZS-NTTA-Extension.git

# 7. 检查分支名称
git branch

# 8. 如果远程仓库不为空，先拉取合并
git pull origin master --allow-unrelated-histories

# 9. 如果有冲突，解决后提交
git add .
git commit -m "Merge remote and local repositories"

# 10. 开启VPN（如果在中国大陆）

# 11. 推送到远程仓库
git push -u origin master
```

## 注意事项
1. **网络问题**：在中国大陆访问GitHub通常需要VPN或代理
2. **分支名称**：注意检查本地分支名称（master/main）与远程仓库匹配
3. **远程仓库内容**：如果远程仓库已有内容，需要先拉取合并
4. **合并冲突**：遇到冲突时要手动解决，通常是README.md文件冲突
5. **首次推送**：可能需要较长时间，取决于项目大小和网络速度
6. **敏感信息**：确保不要提交敏感信息（密码、API密钥等）
7. **备份**：建议定期备份重要代码
8. **分支管理**：可以使用分支进行功能开发，避免直接在main分支上工作