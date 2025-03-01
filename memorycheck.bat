@echo off
setlocal enabledelayedexpansion

rem 初始化计数器(4小时=240分钟)
set /a max_count=240
set /a count=0

:loop
rem 退出条件判断
if %count% equ %max_count% (
    echo 已完成4小时内存监控 >> memory_log.txt
    timeout /t 3 > nul
    exit
)

rem 获取标准化时间戳
for /f "tokens=2 delims==" %%a in ('wmic path win32_operatingsystem get LocalDateTime /value 2^>nul') do (
    set "datetime=%%a"
)
set "logtime=!datetime:~0,4!-!datetime:~4,2!-!datetime:~6,2! !datetime:~8,2!:!datetime:~10,2!:!datetime:~12,2!"

rem 获取内存信息（单位：KB）
for /f "tokens=1-2 delims==" %%a in ('wmic OS get FreePhysicalMemory^,TotalVisibleMemorySize /value 2^>nul ^| findstr "FreePhysicalMemory TotalVisibleMemorySize"') do (
    set "%%a=%%b"
)

rem 计算内存使用情况
set /a TotalKB   = !TotalVisibleMemorySize!
set /a FreeKB    = !FreePhysicalMemory!
set /a UsedKB    = !TotalKB! - !FreeKB!

rem 转换为GB并保留两位小数
set /a TotalGBx100  = !TotalKB! * 100 / 1048576
set TotalGB=!TotalGBx100:~0,-2!.!TotalGBx100:~-2!

set /a UsedGBx100   = !UsedKB! * 100 / 1048576
set UsedGB=!UsedGBx100:~0,-2!.!UsedGBx100:~-2!

set /a FreeGBx100   = !FreeKB! * 100 / 1048576
set FreeGB=!FreeGBx100:~0,-2!.!FreeGBx100:~-2!

rem 计算使用百分比
set /a UsedPercent = !UsedKB! * 100 / !TotalKB!

rem 写入日志文件
echo [!logtime!] 总量: !TotalGB! GB  已用: !UsedGB! GB (!UsedPercent!%%)  剩余: !FreeGB! GB >> memory_log.txt

rem 等待60秒
echo 进度：[!count!/%max_count%] 下次记录时间：!time!
timeout /t 60 /nobreak > nul

rem 计数器递增
set /a count+=1

goto loop