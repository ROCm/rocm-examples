param(
    [Parameter(Mandatory)]
    [string]$Path = "Debug",
    [string]$Filter = "*.exe",
    [int]$Timeout = 10,
    [string[]]$Skipped = @()
)
$Skipped = $Skipped | ForEach-Object { $_.Trim() }

Write-Host "Testing all '$Filter' in '$Path' with a timeout of $Timeout"
Write-Host "Skipping the following examples:"
foreach($item in $Skipped) {
    Write-Host "- $item"
}

$FailureCount = 0
$Results = @()

Get-ChildItem -Recurse -Path $Path -Filter $Filter -Exclude $Skipped | ForEach-Object {
    Write-Host ("`e[36m-- {0}`e[0m" -f $_.Name)
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
    } -ArgumentList $_.FullName

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
        $FailureCount += 1
    }

    # Put into a hash table and append to a list for table magic!
    $Results += [PSCustomObject]@{
        Name       = $_.Name
        Pass       = $Status
        ExitStatus = $ExitDisplay
        Time       = $TimeSpan
    }

    # Clean up!
    Remove-Job -force $Job
}

$Results | Format-Table

if ($FailureCount -gt 0) {
    throw "$FailureCount failed jobs!"
}
