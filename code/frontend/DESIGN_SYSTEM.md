# 设计规范 —— 对齐 Notion 设计语言

> 本项目前端视觉对齐 Notion 设计语言，核心原则：**简洁、现代、专业、克制**。
> 实现载体：`src/style.css` 中的 CSS 变量（Design Tokens），组件统一引用变量，不写死色值。

## 1. 色彩体系

| 角色 | Token | 色值 | 说明 |
|------|-------|------|------|
| 主色 | `--color-accent` | `#0075de` | Notion Blue，唯一高饱和强调色，用于 CTA/链接/选中态 |
| 主色悬停 | `--color-accent-hover` | `#005bab` | 按下/悬停态更深蓝 |
| 主色浅底 | `--color-accent-soft` | `rgba(0,117,222,.08)` | hover/选中背景 |
| 品牌次级 | `--color-brand-deep` | `#213183` | 深海军蓝，少量使用 |
| 主文字 | `--color-ink` | `rgba(0,0,0,.95)` | 近黑（非纯黑，营造纸感） |
| 正文 | `--color-ink-soft` | `#37352f` | 暖深灰 |
| 次要 | `--color-charcoal` | `#615d59` | 次要说明 |
| 三级 | `--color-slate` | `#a39e98` | 辅助说明 |
| 占位 | `--color-muted` | `#c4c0bb` | 占位/禁用 |
| 背景 | `--color-bg` | `#ffffff` | 页面/卡片主背景（纯白） |
| 浅背景 | `--color-bg-soft` | `#f9f9f8` | 分区浅背景（暖白） |
| 安静背景 | `--color-bg-quiet` | `#f7f7f5` | 更安静的分区 |
| 暖灰背景 | `--color-bg-warm` | `#f6f5f4` | hover/选中底 |
| 边框 | `--color-border` | `rgba(0,0,0,.11)` | 超细 1px 分隔线 |
| 成功 | `--color-success` | `#0f7b6c` | 语义绿 |
| 警告 | `--color-warning` | `#ad5a00` | 语义橙 |
| 错误 | `--color-error` | `#eb5757` | 语义红 |

**原则**：暖中性基色（暖灰而非冷灰）+ 单一蓝强调，绝不使用多彩渐变、冷灰堆砌。

## 2. 字体排版

- **字体家族**：`--font-sans` = Inter + 系统字体栈 + `PingFang SC`/`Microsoft YaHei`（中文兜底）
- **字号层级**：display 40 / h3 36 / h2 28 / h1 22 / 正文 16 / 辅助 14 / 小字 12（px）
- **字重层级**：400 正文 · 500 UI/按钮 · 600 半粗标签 · 700 展示标题
- **行高**：正文 1.55（长文可读），标题 1.2（紧凑）
- **字间距**：展示标题负字间距 `-0.02em`（Notion 紧凑感来源）

## 3. 布局栅格与留白

- **基格**：4px 体系，`--space-1`(4px) ~ `--space-9`(96px)
- **容器**：`--container-max: 1280px`，居中，12 列栅格，`--gutter: 24px`
- **留白**：桌面宽幅内容区块，手机单栏，始终维持充足留白

## 4. 圆角 / 阴影 / 动效

- **圆角**：`--radius-sm`(4) / `--radius-md`(8) / `--radius-lg`(12) / `--radius-pill`(999)
- **阴影**：多层低透明度（累计 ≤0.05），深度"被感知而非被看见"
- **缓动**：`--ease: cubic-bezier(0.645, 0.045, 0.355, 1)`
- **时长**：`--duration-micro`(200ms) / `--duration-small`(400ms) / `--duration-medium`(800ms)

## 5. 响应式断点

- `--bp-tablet: 1024px`（降栅格）
- `--bp-mobile: 768px`（单栏）

## 6. 图标 / 插画 / 动效选用原则

- **图标**：线性/扁平，单色，跟随文字颜色（`currentColor`），不加彩
- **插画**：以 emoji 作为轻量"插画"（Notion 的做法），不用复杂插画
- **动效**：克制的微交互——hover 背景色过渡、focus 蓝色 ring、按钮按下变深，不做过场动画
- **禁用**：多彩渐变背景、重阴影、霓虹 glow、装饰性 serif 字体

## 落地清单

- [x] `src/style.css`：完整 Design Tokens + 全局基础样式
- [x] `src/App.vue`：背景去渐变、aurora 降为极淡、按钮/logo 改纯蓝、圆角/阴影对齐
- [x] `src/KbChat.vue`：硬编码色值批量替换为 tokens
- [x] `index.html`：标题更新 + Inter 字体引入说明（注释）
