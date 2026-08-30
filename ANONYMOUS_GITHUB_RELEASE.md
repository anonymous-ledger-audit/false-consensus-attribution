# Anonymous GitHub release checklist

Use a fresh local Git history for the sanitized package. Do not copy a `.git`
directory from a working research folder.

## 1. Verify the package

From the extracted `reproducibility-package` directory:

```powershell
python tools/verify_release.py
Get-ChildItem -Recurse -Force | Select-Object FullName
```

Confirm that the verifier passes and that no unplanned file is present.

## 2. Create an unlinked commit identity

```powershell
git init
git config --local user.name "Anonymous Authors"
git config --local user.email "anonymous-authors@example.invalid"
git config --local --get user.name
git config --local --get user.email
```

The address above is deliberately invalid and should not be added to any
GitHub account. Local configuration overrides the machine-wide Git identity
for this repository.

## 3. Commit the release

```powershell
git add .
git status --short
git commit -m "Anonymous reproducibility package"
git branch -M main
git log -1 --format=fuller
```

Inspect the staged inventory before committing. The final log must show
`Anonymous Authors <anonymous-authors@example.invalid>` for both author and
committer.

## 4. Push to the private review repository

```powershell
git remote add origin https://github.com/anonymous-ledger-audit/reproducibility-package.git
git push -u origin main
```

Authentication may use the managing account; the public commit identity is
determined by the local author and committer fields above.

## 5. Audit before changing visibility

- Confirm the organization membership is private.
- Confirm the repository contains only the intended release files.
- Open the commit page and verify that the commit is not linked to a personal
  profile.
- Check the repository page while signed out or in a private browser window.
- Keep Issues, Discussions, Projects, and Wikis disabled unless required.
- Change the repository to public only after these checks pass.
