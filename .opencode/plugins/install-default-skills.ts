import type { Plugin } from "@opencode-ai/plugin"
import * as fs from "fs"
import * as path from "path"
import * as os from "os"
import crypto from "crypto"
import { execFileSync } from "child_process"

const logFile = path.join(__dirname, "install_error.log")

const REPO_URL = "https://gitcode.com/cann-agent/skills.git"
const DEFAULT_SKILLS = ["gitcode-pr", "gitcode-issue", "api-doc-generator", "gitcode-pipeline"]
const CLONE_TIMEOUT_MS = 20000

function log(message: string) {
  const timestamp = new Date().toISOString()
  fs.appendFileSync(logFile, `[${timestamp}] ${message}\n`)
}

function getFileHash(filePath: string): string | null {
  if (!fs.existsSync(filePath)) return null
  try {
    const content = fs.readFileSync(filePath, 'utf-8')
    return crypto.createHash('md5').update(content).digest('hex')
  } catch (error) {
    return null
  }
}

interface SkillState {
  name: string
  exists: boolean
  hash: string | null
}

function findGitRoot(startDir: string): string {
  let dir = startDir
  while (dir !== path.dirname(dir)) {
    if (fs.existsSync(path.join(dir, ".git"))) {
      return dir
    }
    dir = path.dirname(dir)
  }
  return startDir
}

function createSkillLink(targetPath: string, linkPath: string): void {
  fs.rmSync(linkPath, { recursive: true, force: true })
  if (process.platform === 'win32') {
    fs.symlinkSync(targetPath, linkPath, 'junction')
  } else {
    const relTarget = path.relative(path.dirname(linkPath), targetPath)
    fs.symlinkSync(relTarget, linkPath)
  }
}

function ensureGitignore(gitignorePath: string): void {
  let content = ""
  if (fs.existsSync(gitignorePath)) {
    content = fs.readFileSync(gitignorePath, 'utf-8')
  }
  const lines = content.split('\n')
  let changed = false
  for (const skill of DEFAULT_SKILLS) {
    const entry = `.claude/skills/${skill}`
    if (!lines.includes(entry)) {
      if (content.length > 0 && !content.endsWith('\n')) {
        content += '\n'
      }
      content += entry + '\n'
      lines.push(entry)
      changed = true
    }
  }
  if (changed) {
    fs.writeFileSync(gitignorePath, content)
  }
}

function cloneSkillsRepo(tmpRepo: string): void {
  execFileSync('git', ['clone', '--depth', '1', REPO_URL, tmpRepo], {
    timeout: CLONE_TIMEOUT_MS,
    stdio: 'pipe'
  })
}

function installSkillsToRemote(rootDir: string): void {
  const skillsDir = path.join(rootDir, ".claude", "skills")
  const remoteDir = path.join(skillsDir, "_remote")
  fs.mkdirSync(remoteDir, { recursive: true })

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "skills-install-"))
  try {
    const tmpRepo = path.join(tmpDir, "skills")
    cloneSkillsRepo(tmpRepo)

    const repoSkillsDir = path.join(tmpRepo, "skills")
    if (!fs.existsSync(repoSkillsDir)) {
      throw new Error(`skills directory not found in repository: ${repoSkillsDir}`)
    }

    for (const skill of DEFAULT_SKILLS) {
      const src = path.join(repoSkillsDir, skill)
      if (!fs.existsSync(src)) {
        log(`Skill '${skill}' not found in repository`)
        continue
      }
      const dest = path.join(remoteDir, skill)
      fs.rmSync(dest, { recursive: true, force: true })
      fs.cpSync(src, dest, { recursive: true })
      createSkillLink(dest, path.join(skillsDir, skill))
    }

    ensureGitignore(path.join(rootDir, ".gitignore"))
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true })
  }
}

export const InstallSkillsPlugin: Plugin = async ({ $, directory }) => {
  const rootDir = findGitRoot(directory)
  const installSkills = async () => {
    try {
      // 记录安装前四个skill文件的状态
      const skillsToCheck = DEFAULT_SKILLS.map(name => ({
        name,
        path: path.join(rootDir, ".claude", "skills", name, "SKILL.md")
      }))

      const beforeStates: SkillState[] = skillsToCheck.map(skill => ({
        name: skill.name,
        exists: fs.existsSync(skill.path),
        hash: getFileHash(skill.path)
      }))

      installSkillsToRemote(rootDir)

      // 记录安装后四个skill文件的状态
      const afterStates: SkillState[] = skillsToCheck.map(skill => ({
        name: skill.name,
        exists: fs.existsSync(skill.path),
        hash: getFileHash(skill.path)
      }))

      // 检查是否有任何skill发生了变化
      let hasChanges = false
      const changedSkills: string[] = []

      for (let i = 0; i < beforeStates.length; i++) {
        const before = beforeStates[i]
        const after = afterStates[i]

        if (!before.exists && after.exists) {
          hasChanges = true
          changedSkills.push(`${before.name} 新安装`)
        } else if (before.exists && after.exists && before.hash !== after.hash) {
          hasChanges = true
          changedSkills.push(`${before.name} 已更新`)
        }
      }

      // 只有当所有skill都在安装前后完全相同时才不打印提示
      if (hasChanges && changedSkills.length > 0) {
        setTimeout(() => {
        process.stdout.write(`💡 ${changedSkills.join(', ')}，重启opencode才能完全生效\n\n`)
        }, 1000)
      }
} catch (err) {
      const error = err as Error & { stderr?: Buffer }
      log(`Command failed: ${error.message}`)
      const stderrStr = error.stderr ? error.stderr.toString() : ""
      if (stderrStr) log(`stderr from error: ${stderrStr}`)
      const errorMarkerPath = path.join(rootDir, ".opencode_skills_error")
      let detail = ""
      if (error.message && error.message.includes("timed out")) {
        detail = `网络连接超时，无法访问远程仓库。请检查网络连接后重试。\n${error.message}`
      } else {
        detail = stderrStr ? `${error.message}\n${stderrStr}` : error.message
      }
      const errorMessage = `❌ 安装默认技能时出错了，请输入指令"安装默认skill"重新安装\n错误详情: ${detail}\n`
      fs.writeFileSync(errorMarkerPath, errorMessage)

      // 延迟打印到标准输出，避免被界面刷新清除
      setTimeout(() => {
        process.stdout.write(`❌ 安装默认技能时出错了，请输入指令“安装默认skill”重新安装\n`)
        process.stdout.write(`   错误详情: ${detail}\n`)
        process.stdout.write(`   错误详情请查看: ${errorMarkerPath}\n\n`)
      }, 2000)
    }
  };

  installSkills();
  return {
    event: async ({ event }) => {}
  }
}
