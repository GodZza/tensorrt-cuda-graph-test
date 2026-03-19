# CUDA 12.8 + Visual Studio 构建问题解决过程

## 问题背景

在使用 Visual Studio 2019/2022 编译 CUDA 12.8 项目时，遇到编译失败的问题。

## 问题分析

### 问题 1：CUDA 12.8 不支持 compute_52 架构

**现象**：
```
error MSB3721: nvcc.exe ... -gencode=arch=compute_52,code="sm_52,compute_52" ...
```

**原因**：
- CUDA 12.8 移除了对 `compute_52` (Maxwell 架构) 的支持
- Visual Studio 的 CUDA 集成文件 (`CUDA 12.8.props`) 中硬编码了默认架构 `compute_52,sm_52`

**解决方案**：
修改 `CUDA 12.8.props` 文件，将默认架构从 `compute_52,sm_52` 改为 `compute_75,sm_75`

**文件位置**：
- VS2019: `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations\CUDA 12.8.props`
- VS2022: `C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 12.8.props`

**修改命令**：
```powershell
# VS2019
powershell -Command "(Get-Content 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations\CUDA 12.8.props') -replace 'compute_52,sm_52', 'compute_75,sm_75' | Set-Content 'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations\CUDA 12.8.props'"

# VS2022
powershell -Command "(Get-Content 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 12.8.props') -replace 'compute_52,sm_52', 'compute_75,sm_75' | Set-Content 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations\CUDA 12.8.props'"
```

### 问题 2：Visual Studio CUDA 集成的 -gencode 参数引号转义问题

**现象**：
即使修改了 props 文件，CMake 在检测 CUDA 编译器时仍然失败：
```
-gencode=arch=compute_75,code=\"sm_75,compute_75\"
```
引号转义有问题，导致 nvcc 无法正确解析参数。

**原因**：
- Visual Studio 的 CUDA 集成在处理 `-gencode` 参数时，引号转义方式与 nvcc 不兼容
- 这是 VS2022 + CUDA 12.8 的已知兼容性问题

**尝试的解决方案**：

| 方案 | 结果 |
|------|------|
| 使用 VS2019 生成器 | 失败，同样的问题 |
| 添加 `-allow-unsupported-compiler` 标志 | 失败，问题在编译器 ID 检测阶段 |
| 设置 `CMAKE_CUDA_FLAGS` | 失败，VS 集成会覆盖 |
| 使用 Ninja 生成器 | 失败，Ninja 未正确安装 |
| **使用 NMake Makefiles 生成器** | ✅ 成功 |

### 问题 3：VS2019 和 VS2022 环境冲突

**现象**：
```
The input line is too long.
The syntax of the command is incorrect.
```

**原因**：
- 系统同时安装了 VS2019 和 VS2022
- PATH 环境变量中两个版本的 cl.exe 冲突
- 调用 `vcvars64.bat` 时命令行过长

**解决方案**：
使用 NMake Makefiles 生成器，不依赖 Visual Studio 的 CUDA 集成。

## 最终解决方案

使用 **NMake Makefiles** 生成器绕过 Visual Studio 的 CUDA 集成问题：

```batch
cmake -G "NMake Makefiles" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES=75;86;89 ^
    -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" ^
    -DTensorRT_ROOT_DIR="C:/TensorRT/TensorRT-10.9.0.34" ^
    -DCMAKE_C_COMPILER=cl ^
    -DCMAKE_CXX_COMPILER=cl ^
    ..

nmake
```

## 为什么 NMake 能成功？

| 对比项 | Visual Studio 生成器 | NMake 生成器 |
|--------|---------------------|--------------|
| CUDA 编译方式 | 通过 VS 的 CUDA 集成（.props/.targets） | 直接调用 nvcc |
| 架构参数传递 | 通过 `-gencode` 参数（引号转义问题） | 通过 `-arch` 参数 |
| 编译器 ID 检测 | 使用 VS 的默认配置 | 直接使用 CMake 配置 |
| 兼容性 | 依赖 VS 和 CUDA 版本兼容性 | 只依赖 nvcc 和 MSVC |

## 经验总结

1. **CUDA 12.8 不再支持旧架构**：最低支持 `compute_75` (Turing)
2. **VS 的 CUDA 集成可能有兼容性问题**：特别是新版本的 CUDA 和旧版本的 VS
3. **NMake 是可靠的备选方案**：绕过 VS 集成，直接使用 nvcc
4. **多版本 VS 共存可能有问题**：环境变量冲突

## 相关文件

- `build_nmake.bat` - NMake 构建脚本
- `build.bat` - Visual Studio 构建脚本（有问题）
- `fix_cuda_arch.bat` - 修复 CUDA props 文件的脚本
- `fix_cuda_arch_vs2019.bat` - 修复 VS2019 CUDA props 文件的脚本
