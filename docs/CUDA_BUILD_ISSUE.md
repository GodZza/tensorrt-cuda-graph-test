# CUDA 12.8 构建问题说明

## 问题描述

在使用 Visual Studio 2019/2022 编译 CUDA 12.8 项目时，会遇到以下错误：

```
error MSB3721: nvcc.exe ... -gencode=arch=compute_52,code="sm_52,compute_52" ...
```

## 根本原因

### 1. CUDA 12.8 移除了对旧架构的支持

CUDA 12.8 不再支持 `compute_52` (Maxwell 架构，GTX 900 系列) 及更早的架构。

支持的最低架构是 `compute_75` (Turing 架构，RTX 20 系列)。

### 2. Visual Studio CUDA 集成的默认配置问题

Visual Studio 的 CUDA 集成文件 (`CUDA 12.8.props`) 中硬编码了默认架构：

```
文件位置：
VS2019: C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations\CUDA 12.8.props
VS2022: C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 12.8.props

默认值（第 113 行左右）：
<CodeGeneration>compute_52,sm_52</CodeGeneration>
```

这个默认值 `compute_52,sm_52` 在 CUDA 12.8 中已经无效，导致编译失败。

### 3. CMake 编译器 ID 检测

CMake 在配置阶段会检测 CUDA 编译器，此时会使用 Visual Studio 的默认配置，即使你在 CMakeLists.txt 中设置了 `CMAKE_CUDA_ARCHITECTURES`，CMake 的编译器 ID 检测阶段仍然会使用 VS 的默认值。

## 解决方案

### 方案 1：修改 CUDA 12.8.props 文件（推荐）

将默认架构从 `compute_52,sm_52` 改为 `compute_75,sm_75`：

```powershell
# VS2019
powershell -Command "(Get-Content 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations\CUDA 12.8.props') -replace 'compute_52,sm_52', 'compute_75,sm_75' | Set-Content 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations\CUDA 12.8.props'"

# VS2022
powershell -Command "(Get-Content 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 12.8.props') -replace 'compute_52,sm_52', 'compute_75,sm_75' | Set-Content 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 12.8.props'"
```

### 方案 2：升级到 CUDA 13.x

CUDA 13.x 已经修复了这个问题，默认使用 `compute_75`。

### 方案 3：降级到 CUDA 12.6 或更早版本

CUDA 12.6 及更早版本仍然支持 `compute_52`。

## 架构对照表

| 架构代号 | GPU 系列 | 示例显卡 |
|---------|---------|---------|
| compute_52 | Maxwell | GTX 900, GTX Titan X |
| compute_60 | Pascal | GTX 1000 |
| compute_70 | Volta | Titan V |
| compute_75 | Turing | RTX 2000, GTX 1600 |
| compute_80 | Ampere | RTX 3000, A100 |
| compute_86 | Ampere | RTX 3000 Mobile |
| compute_89 | Ada Lovelace | RTX 4000 |
| compute_90 | Hopper | H100 |

## CUDA 12.8 支持的架构

```
compute_75, compute_80, compute_86, compute_87, compute_88, compute_89,
compute_90, compute_90a, compute_100, compute_100f, compute_100a,
compute_103, compute_103f, compute_103a, compute_110, compute_110f,
compute_110a, compute_120, compute_120f, compute_120a,
compute_121, compute_121f, compute_121a
```

## 注意事项

1. 修改系统文件需要管理员权限
2. 修改后需要重新运行 CMake 配置
3. 建议备份原始文件（修改脚本会自动处理）
