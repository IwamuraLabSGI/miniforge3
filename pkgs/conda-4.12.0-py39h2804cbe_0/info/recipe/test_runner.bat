:: Restrict scope of all environment variables
setlocal
verify >nul

:: Deactivate external conda.
call %PREFIX%\condabin\deactivate.bat
if errorlevel 1 exit 1

:: Configure special conda directories and files.
set "CONDARC=%PREFIX%\.condarc"
set "CONDA_ENVS_DIRS=%PREFIX%\envs"
set "CONDA_PKGS_DIRS=%PREFIX%\pkgs"

:: Add stubs for special conda directories and files.
copy nul %CONDARC%
if errorlevel 1 exit 1
mkdir %CONDA_ENVS_DIRS%
if errorlevel 1 exit 1
mkdir %CONDA_PKGS_DIRS%
if errorlevel 1 exit 1

:: Activate the built conda.
conda activate base
if errorlevel 1 exit 1

:: Run conda tests.
call %CD%\commands_to_test.bat
if errorlevel 1 exit 1

:: Deactivate the built conda when done.
:: Not necessary, but a good test.
conda deactivate
if errorlevel 1 exit 1

endlocal
