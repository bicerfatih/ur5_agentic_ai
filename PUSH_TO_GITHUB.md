# Push to GitHub (bicerfatih / ur5_agentic_ai)

Local commit is on branch `main`. Remote: `git@github.com:bicerfatih/ur5_agentic_ai.git`

## 1. Add SSH key (one time)

Open: https://github.com/settings/keys → **New SSH key**

Paste this public key:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIK0ARUp53I4MAekv42xkFKAlNRZWdh3CZvAQxgTunYta fatihbicer@github-ur5-agentic
```

## 2. Create empty private repo

https://github.com/new

- Name: `ur5_agentic_ai`
- Visibility: **Private**
- Do **not** add README, .gitignore, or license (repo must be empty)

## 3. Push

```bash
cd ~/Downloads/ur5_agentic_ai
git push -u origin main
```

Test SSH first: `ssh -T git@github.com` → should say "Hi bicerfatih!"
