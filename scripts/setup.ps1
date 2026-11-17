param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]] $Arguments
)
$ErrorActionPreference = 'Stop'
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
& (Get-Command python).Source (Join-Path $scriptRoot 'setup.py') @Arguments
exit $LASTEXITCODE
