param(
    [Parameter(Mandatory)]
    [string]$Path = "Debug",
    [string]$Filter = "*.exe",
    [int]$Timeout = 60,
    [string[]]$Skip = @()
)
$Skip = $Skip | ForEach-Object { $_.Trim() }

Write-Host "Testing all '$Filter' in '$Path' with a timeout of $Timeout"
Write-Host "Skipping examples that match any of:"
foreach($item in $Skip) {
    Write-Host "- $item"
}

$FailureCount = 0
$Results = @()

function Run-Example {
    param(
        [System.IO.FileInfo]$FileInfo
    )

    $Job = Start-Job -ScriptBlock {
        param([string]$FullName)
        $Time = Measure-Command { 
            try {
                $Log = & $FullName
                $JobExitStatus = $LASTEXITCODE
            } catch {
                $JobExitStatus = "CRASH!"
            }
        }
        return [PSCustomObject]@{
            ExitStatus = $JobExitStatus
            Log        = $Log
            Time       = $Time
        }
    } -ArgumentList $FileInfo.FullName

    # Execute the job with a timeout
    $Job | Wait-Job -TimeOut $Timeout | Out-Null

    # Get the results from the job!
    $Result = Receive-Job $Job
    Write-Host $Result.Log

    if ($null -ne $Result.ExitStatus) {
        $TimeSpan   = $Result.Time.toString("mm\:ss\.fff")
        $ExitStatus = $Result.ExitStatus
    } else {
        $ExitStatus = "Timeout!"
        $TimeSpan   = $null
    }

    if ($Result.ExitStatus -eq 0) {
        # Exited gracefully!
        $Status = "`e[32mPass`e[0m"
        $ExitDisplay = "`e[32m$ExitStatus`e[0m"
    } else {
        $ExitDisplay = "`e[31m$ExitStatus`e[0m"
        
        # Otherwise, fail!
        $Status = "`e[31m`e[1mFail`e[0m"
        $script:FailureCount += 1
    }

    # Clean up!
    Remove-Job -force $Job

    [PSCustomObject]@{
        Name       = $FileInfo.Name
        State      = $Status
        ExitStatus = $ExitDisplay
        Time       = $TimeSpan
    }
}

Get-ChildItem -Recurse -File -Path $Path -Filter $Filter | ForEach-Object {
    Write-Host ("`e[36m-- {0}`e[0m" -f $_.Name)

    $ShouldSkip = $false
    foreach($F in $Skip) {
        if ($_.Name -like $F) {
            Write-Host "`e[33m`e[1mSkipped by wildcard:`e[0m $F"
            $ShouldSkip = $true
            break
        }
    }

    # Put into a hash table and append to a list for table magic!
    if (-not $ShouldSkip) {
        $Results += Run-Example $_
    } else {
        $Results += [PSCustomObject]@{
            Name       = $_.Name
            State      = "`e[33m`e[1mSkip`e[0m"
            ExitStatus = $null 
            Time       = $null
        }
    }
}

$Results | Format-Table

if ($FailureCount -gt 0) {
    throw "$FailureCount failed jobs!"
}
