# NMake 构建工具说明

## 什么是 NMake？

NMake (Microsoft Program Maintenance Utility) 是 Microsoft Visual Studio 自带的命令行构建工具，类似于 Unix 的 `make`。

## 为什么 NMake 能成功？

### Visual Studio 生成器 vs NMake 生成器

| 对比项 | Visual Studio 生成器 | NMake 生成器 |
|--------|---------------------|--------------|
| **CUDA 编译方式** | 通过 VS 的 CUDA 集成（.props/.targets 文件） | 直接调用 nvcc 编译器 |
| **架构参数传递** | 使用 `-gencode=arch=compute_75,code="sm_75,compute_75"` | 使用 `-arch=sm_75` |
| **编译器 ID 检测** | 使用 VS 的默认配置（可能有问题） | 直接使用 CMake 配置 |
| **中间文件处理** | VS 集成控制，可能有引号转义问题 | nvcc 直接处理 |
| **依赖关系** | 依赖 VS 和 CUDA 版本兼容性 | 只依赖 nvcc 和 MSVC 编译器 |

### 核心区别

**Visual Studio 生成器**：
1. 生成 `.sln` 和 `.vcxproj` 文件
2. 通过 MSBuild 调用编译器
3. CUDA 编译通过 VS 的 CUDA 集成（`CUDA 12.8.props`/`.targets`）
4. 参数传递经过 VS 处理，可能有引号转义问题

**NMake 生成器**：
1. 生成 `Makefile` 文件
2. 直接调用 `nmake` 命令
3. CUDA 编译直接调用 `nvcc`，参数直接传递
4. 不经过 VS 的 CUDA 集成，避免了兼容性问题

### 为什么 NMake 绕过了问题？

1. **不使用 VS CUDA 集成**：NMake 直接调用 nvcc，不经过 `CUDA 12.8.props` 文件
2. **参数直接传递**：CMake 生成的 Makefile 中，CUDA 参数直接传递给 nvcc
3. **架构参数简化**：使用 `-arch=sm_75` 而不是复杂的 `-gencode` 参数
4. **编译器检测简单**：CMake 直接检测 nvcc，不经过 VS 的配置

## NMake 基本用法

```batch
# 配置项目
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..

# 编译
nmake

# 清理
nmake clean

# 并行编译
nmake /N
```

## NMake 的优缺点

### 优点
- 绕过 VS CUDA 集成的兼容性问题
- 编译速度快
- 参数传递简单直接
- 适合 CI/CD 环境

### 缺点
- 没有 VS 的图形化调试支持
- 需要手动配置环境（调用 vcvars64.bat）
- 不生成 VS 项目文件

## 总结

NMake 成功的关键在于：**直接调用 nvcc，不经过 Visual Studio 的 CUDA 集成**，从而避免了 VS CUDA 集成中的架构参数引号转义问题。
