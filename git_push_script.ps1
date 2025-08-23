<#
.SYNOPSIS
    A script to automate creating a new Git branch, committing, and pushing.
.DESCRIPTION
    This script simplifies the process of creating a new feature branch, adding all current changes,
    committing them with a message, and preparing to push to the remote repository.
    It should be run from the root of the Git repository.
.PARAMETER BranchName
    The name of the new branch to create.
.PARAMETER CommitMessage
    The commit message for the changes.
.EXAMPLE
    .\git_push_script.ps1 -BranchName "feature/new-ui" -CommitMessage "Add new UI components"
#>
param (
    [Parameter(Mandatory=$true)]
    [string]$BranchName,
    [Parameter(Mandatory=$true)]
    [string]$CommitMessage
)

# 1. Create a new branch
git checkout -b $BranchName

# 2. Add all files to the staging area
git add .

# 3. Commit the changes
git commit -m $CommitMessage

# 4. Push the branch to GitHub
Write-Host "Branch '$BranchName' is ready to be pushed. Run the following command to push:"
Write-Host "git push -u origin $BranchName" -ForegroundColor Green

# 5. Display status
git status
