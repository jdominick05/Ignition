#!/usr/bin/env bash
# Commit staged (or given) files with this repo's required hygiene checks and
# attribution trailer. Never pushes -- push stays a separate, confirmed step.
#
#   ./scripts/commit.sh --subject "..." --body-file /tmp/body.txt \
#       --session-url https://claude.ai/code/session_XXXX \
#       results/foo.log README.md
#
#   ./scripts/commit.sh -m "..." -F body.txt -s <url>   # short flags, uses
#                                                        # whatever is already staged
#
# AGY_SESSION_URL or CLAUDE_SESSION_URL can supply --session-url.
#
# Checks before committing, all restricted to files under results/ (if present):
#   - no embedded NUL bytes -- PowerShell's `*>` writes UTF-16, which makes
#     every later grep/rg silently match nothing
#   - no literal local profile path -- replace with C:\Users\<user> by hand
#     before staging, this script only detects it, it does not rewrite logs
#
# Positional args (if any) are `git add`-ed by name -- never -A, never `.`.
# With none, whatever is already staged is committed as-is.

. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SUBJECT="" BODY_FILE="" SESSION_URL="${AGY_SESSION_URL:-${CLAUDE_SESSION_URL:-}}"
SESSION_TRAILER="${SESSION_TRAILER:-Codex-Session}"
COAUTHOR="${COAUTHOR:-Codex <noreply@openai.com>}"
FILES=()

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--subject)         SUBJECT="$2"; shift ;;
        -F|--body-file)       BODY_FILE="$2"; shift ;;
        -s|--session-url)     SESSION_URL="$2"; shift ;;
        --session-trailer)    SESSION_TRAILER="$2"; shift ;;
        -c|--coauthor)        COAUTHOR="$2"; shift ;;
        -h|--help)            usage "${BASH_SOURCE[0]}"; exit 0 ;;
        --)                   shift; while [ $# -gt 0 ]; do FILES+=("$1"); shift; done; continue ;;
        -*)                   die "unknown flag $1" ;;
        *)                    FILES+=("$1") ;;
    esac
    shift
done

[ -n "$SUBJECT" ] || die "need --subject/-m"

if [ "${#FILES[@]}" -gt 0 ]; then
    step "staging ${#FILES[@]} file(s)"
    git add -- "${FILES[@]}"
fi

git diff --cached --quiet && die "nothing staged -- pass files to commit, or git add first"

step "checking staged logs are UTF-8 and profile-scrubbed"
UNAME="${USERNAME:-${USER:-}}"
BAD=0
while IFS= read -r f; do
    case "$f" in *.log) ;; *) continue ;; esac
    [ -f "$f" ] || continue
    if ! python -c "import sys; sys.exit(1 if b'\x00' in open(sys.argv[1],'rb').read() else 0)" "$f"; then
        warn "$f looks UTF-16 (embedded NUL bytes) -- decode to UTF-8 before committing"
        BAD=1
    fi
    if [ -n "$UNAME" ]; then
        win_pat="$(printf 'Users\\%s\\' "$UNAME")"
        posix_pat="Users/$UNAME/"
        if grep -qF "$win_pat" "$f" 2>/dev/null || grep -qF "$posix_pat" "$f" 2>/dev/null; then
            warn "$f still has the local profile path -- replace with C:\\Users\\<user>"
            BAD=1
        fi
    fi
done < <(git diff --cached --name-only)
[ "$BAD" = 0 ] || die "fix the above, then re-stage and re-run"
ok "staged files clean"

TMPMSG="$(mktemp)"
trap 'rm -f "$TMPMSG"' EXIT
{
    printf '%s\n' "$SUBJECT"
    if [ -n "$BODY_FILE" ]; then
        need_file "$BODY_FILE"
        printf '\n'
        cat "$BODY_FILE"
    fi
    if [ -n "$SESSION_URL" ]; then
        printf '\nCo-Authored-By: %s\n%s: %s\n' "$COAUTHOR" "$SESSION_TRAILER" "$SESSION_URL"
    fi
} > "$TMPMSG"

step "committing"
git commit -F "$TMPMSG"
ok "committed, not pushed -- review with 'git log -1', push only after confirming."
