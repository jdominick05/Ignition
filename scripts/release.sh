#!/usr/bin/env bash
# Build, verify, tag, and publish an Ignition release to both GitLab (origin)
# and GitHub (github), attaching the verified binary wheel to both platforms.
#
#   ./scripts/release.sh                  # full release based on pyproject.toml version
#   ./scripts/release.sh --dry-run        # run build and package verification only
#   ./scripts/release.sh --version 0.1.0  # override or explicitly declare version

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

VERSION=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        -v|--version) VERSION="$2"; shift ;;
        --dry-run)    DRY_RUN=1 ;;
        -h|--help)    usage "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown arg $1" ;;
    esac
    shift
done

# 1. Pre-flight hygiene checks
step "running pre-flight repository hygiene checks"
git diff --quiet || die "working tree has uncommitted modifications"
git diff --cached --quiet || die "working tree has staged uncommitted changes"

untracked="$(git ls-files --others --exclude-standard)"
if [ -n "$untracked" ]; then
    die "working tree has untracked files:\n$untracked"
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[ "$BRANCH" = "main" ] || die "releases must be cut from 'main' (currently on '$BRANCH')"

for r in origin github; do
    git remote get-url "$r" >/dev/null 2>&1 || die "remote '$r' is not configured (git remote -v)"
done

step "verifying remotes and synchronization status"
git fetch origin --quiet || die "failed to fetch from origin"
git fetch github --quiet || die "failed to fetch from github"

LOCAL_HEAD="$(git rev-parse HEAD)"
ORIGIN_HEAD="$(git rev-parse origin/main)"
GITHUB_HEAD="$(git rev-parse github/main)"

[ "$LOCAL_HEAD" = "$ORIGIN_HEAD" ] || die "local main ($LOCAL_HEAD) does not match origin/main ($ORIGIN_HEAD); push or pull first"
[ "$LOCAL_HEAD" = "$GITHUB_HEAD" ] || die "local main ($LOCAL_HEAD) does not match github/main ($GITHUB_HEAD); sync remotes first"
ok "local main is clean and synchronized across origin and github ($LOCAL_HEAD)"

# 2. Extract and validate version
if [ -z "$VERSION" ]; then
    VERSION="$(python -c '
import re
with open("pyproject.toml", "r", encoding="utf-8") as f:
    m = re.search(r"version\s*=\s*[\x22\x27]([^\x22\x27]+)[\x22\x27]", f.read())
    if m:
        print(m.group(1))
    else:
        raise RuntimeError("version not found")
')"
fi

[ -n "$VERSION" ] || die "failed to extract version from pyproject.toml"
TAG="v$VERSION"
info "release target: $TAG (version $VERSION)"

if git rev-parse "$TAG" >/dev/null 2>&1; then
    die "tag '$TAG' already exists locally"
fi
if git ls-remote origin "refs/tags/$TAG" | grep -q "$TAG"; then
    die "tag '$TAG' already exists on origin"
fi
if git ls-remote github "refs/tags/$TAG" | grep -q "$TAG"; then
    die "tag '$TAG' already exists on github"
fi
ok "tag '$TAG' is available locally and on all remotes"

# 3. Clean workspace & build distributions
step "cleaning build/dist workspace"
rm -rf build/ dist/ *.egg-info src/*.egg-info

step "building distribution packages (sdist + wheel)"
python -m build || die "python -m build failed"

# 4. Audit packaged wheel assets
step "auditing packaged assets inside wheel archive"
WHEEL_FILE="$(ls -1 dist/ignition_ai-*.whl 2>/dev/null | head -n 1)"
[ -f "$WHEEL_FILE" ] || die "wheel artifact not found in dist/"

python -c "
import sys, zipfile
whl = sys.argv[1]
with zipfile.ZipFile(whl, 'r') as z:
    names = set(z.namelist())
    required = [
        'ignition/assets/im2col_4d_16core.xclbin',
        'ignition/assets/layer_conv0_exec.bin',
        'ignition/assets/layer_conv0_init.bin',
        'ignition/assets/layer_fused_exec.bin',
        'ignition/assets/layer_fused_init.bin',
    ]
    missing = [r for r in required if r not in names]
    if missing:
        sys.exit(f'ERROR: Missing critical assets in wheel: {missing}')

    entry_files = [n for n in names if n.endswith('entry_points.txt')]
    if not entry_files:
        sys.exit('ERROR: entry_points.txt missing in wheel dist-info')
    ep_content = z.read(entry_files[0]).decode('utf-8')
    if 'ignition = ignition.cli.main:cli' not in ep_content:
        sys.exit('ERROR: CLI entrypoint missing from entry_points.txt')

print('  Audit PASSED: im2col_4d_16core.xclbin, transaction templates, and CLI entrypoint verified.')
" "$WHEEL_FILE" || die "wheel asset verification failed"
ok "wheel packaging validated ($WHEEL_FILE)"

if [ "$DRY_RUN" -eq 1 ]; then
    ok "DRY-RUN COMPLETE: Build and asset audits succeeded. Tagging and release dispatch skipped."
    exit 0
fi

# 5. Tag creation
step "creating annotated release tag '$TAG'"
git tag -a "$TAG" -m "Release $TAG"
ok "tag '$TAG' created"

# 6. Push to origin (GitLab) and github (GitHub mirror)
step "pushing tag '$TAG' to remotes"
for r in origin github; do
    step "-> pushing tag to $r"
    if git push "$r" "$TAG"; then
        ok "tag $TAG pushed to $r"
    else
        die "failed to push tag $TAG to $r"
    fi
done

# 7. Multi-platform release artifact publishing
DIST_FILES=(dist/*)

if command -v glab >/dev/null 2>&1; then
    step "creating GitLab release via glab"
    if glab release create "$TAG" "${DIST_FILES[@]}" --name "Release $TAG" --notes "Ignition $TAG: High-Performance AMD Ryzen AI XDNA1 AIE2 Inference Engine."; then
        ok "GitLab release $TAG created successfully"
    else
        warn "glab release creation failed"
    fi
else
    info "glab CLI not detected -- skipping automated GitLab release asset upload"
fi

if command -v gh >/dev/null 2>&1; then
    step "creating GitHub release via gh"
    if gh release create "$TAG" "${DIST_FILES[@]}" --title "Release $TAG" --notes "Ignition $TAG: High-Performance AMD Ryzen AI XDNA1 AIE2 Inference Engine."; then
        ok "GitHub release $TAG created successfully"
    else
        warn "gh release creation failed"
    fi
else
    info "gh CLI not detected -- skipping automated GitHub release asset upload"
fi

ok "release $TAG published successfully to origin and github"
