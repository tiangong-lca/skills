#!/bin/sh
set -eu
umask 077
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PATH

fail() { printf '%s\n' "bootstrap_error:$1" >&2; exit 1; }
hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else fail missing_sha256_tool; fi
}
file_size() { wc -c < "$1" | tr -d '[:space:]'; }
json_string() {
  key=$1
  count=$(grep -c "^[[:space:]]*\"$key\"[[:space:]]*:" "$lock_file" || true)
  [ "$count" = 1 ] || fail "lock_key_$key"
  value=$(sed -n "s/^[[:space:]]*\"$key\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\"[[:space:]]*,\{0,1\}[[:space:]]*$/\\1/p" "$lock_file")
  [ -n "$value" ] || fail "lock_value_$key"
  printf '%s' "$value"
}
json_number() {
  key=$1
  count=$(grep -c "^[[:space:]]*\"$key\"[[:space:]]*:" "$lock_file" || true)
  [ "$count" = 1 ] || fail "lock_key_$key"
  value=$(sed -n "s/^[[:space:]]*\"$key\"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\)[[:space:]]*,\{0,1\}[[:space:]]*$/\\1/p" "$lock_file")
  [ -n "$value" ] || fail "lock_value_$key"
  printf '%s' "$value"
}
validate_sha() { [ "${#1}" = 64 ] && case "$1" in *[!0-9a-f]*) false;; *) true;; esac || fail invalid_sha256; }
validate_relative() {
  value=$1
  [ -n "$value" ] || fail empty_relative_path
  case "$value" in /*|*\\*|*:*|-*|*/-*|.|..|../*|*/../*|*/..|*//*|*./../*) fail unsafe_relative_path;; esac
}
verify_file() {
  file=$1 expected_bytes=$2 expected_sha=$3
  [ -f "$file" ] && [ ! -L "$file" ] || fail missing_regular_file
  [ "$(file_size "$file")" = "$expected_bytes" ] || fail file_size_mismatch
  [ "$(hash_file "$file")" = "$expected_sha" ] || fail file_sha256_mismatch
}
validate_initial_url() {
  case "$1" in
    https://github.com/*/*/releases/download/*/*|https://nodejs.org/dist/v[0-9]*.[0-9]*.[0-9]*/*|https://registry.npmjs.org/*/-/*-[0-9]*.[0-9]*.[0-9]*.tgz) ;;
    *) fail unsafe_distribution_url;;
  esac
  case "$1" in *\?*|*\#*|*' '*) fail mutable_distribution_url;; esac
}
download() {
  download_url=$1 download_target=$2 download_max=$3
  validate_initial_url "$download_url"
  command -v curl >/dev/null 2>&1 || fail missing_curl
  curl --disable --fail --silent --show-error --location --proto '=https' --proto-redir '=https' --tlsv1.2 --max-filesize "$download_max" --output "$download_target" "$download_url" || fail download_failed
}
verify_component_root() {
  component_root=$1 integrity_relative=$2 integrity_sha=$3 expected_count=$4
  validate_relative "$integrity_relative"
  integrity="$component_root/$integrity_relative"
  [ -f "$integrity" ] && [ ! -L "$integrity" ] || fail missing_integrity_file
  [ "$(hash_file "$integrity")" = "$integrity_sha" ] || fail integrity_file_changed
  if find "$component_root" -type l -print | grep . >/dev/null 2>&1; then fail component_contains_link; fi
  count=0
  while IFS= read -r line || [ -n "$line" ]; do
    expected=${line%%  *}; relative=${line#*  }
    [ "$expected" != "$line" ] || fail malformed_checksum_line
    validate_sha "$expected"; validate_relative "$relative"
    [ -f "$component_root/$relative" ] && [ ! -L "$component_root/$relative" ] || fail component_file_missing
    count=$((count + 1))
  done < "$integrity"
  [ "$count" = "$expected_count" ] || fail checksum_count_mismatch
  if command -v sha256sum >/dev/null 2>&1; then (cd "$component_root" && sha256sum -c "$integrity" >/dev/null) || fail component_file_changed
  else (cd "$component_root" && shasum -a 256 -c "$integrity" >/dev/null) || fail component_file_changed; fi
  actual=$(find "$component_root" -type f -print | wc -l | tr -d '[:space:]')
  [ "$actual" = $((expected_count + 1)) ] || fail component_has_extra_file
}
validate_archive() {
  archive=$1 integrity_relative=$2 expected_count=$3 list_file=$4 verbose_file=$5 expected_file=$6
  command -v tar >/dev/null 2>&1 || fail missing_tar
  tar -tzf "$archive" > "$list_file" || fail archive_list_failed
  count=0
  while IFS= read -r relative || [ -n "$relative" ]; do validate_relative "$relative"; count=$((count + 1)); done < "$list_file"
  [ "$count" = $((expected_count + 1)) ] || fail archive_entry_count
  tar -tvzf "$archive" > "$verbose_file" || fail archive_verbose_failed
  while IFS= read -r line || [ -n "$line" ]; do case "$line" in -*) ;; *) fail archive_non_regular_entry;; esac; done < "$verbose_file"
  { awk '{line=$0; sub(/^[0-9a-f][0-9a-f]*  /,"",line); print line}' "$checksum_source"; printf '%s\n' "$integrity_relative"; } | LC_ALL=C sort > "$expected_file"
  LC_ALL=C sort "$list_file" > "$list_file.sorted"
  cmp "$expected_file" "$list_file.sorted" >/dev/null 2>&1 || fail archive_inventory_mismatch
}

script_path=$0
script_dir=$(CDPATH= cd -P -- "$(dirname -- "$script_path")" && pwd -P)
lock_file="$script_dir/bootstrap-lock.json"
[ -f "$lock_file" ] && [ ! -L "$lock_file" ] || fail missing_adjacent_lock
[ "$(json_string schema)" = tiangong-lca.runtime-bootstrap-lock.v1 ] || fail lock_schema
[ "$(json_string bootstrap_protocol)" = tiangong-lca.runtime-bootstrap.v1 ] || fail lock_protocol
script_sha=$(json_string posix_script_sha256); validate_sha "$script_sha"
[ "$(hash_file "$script_path")" = "$script_sha" ] || fail bootstrap_script_changed

os=$(uname -s); arch=$(uname -m)
case "$os:$arch" in
  Darwin:arm64) platform=darwin_arm64;;
  Darwin:x86_64)
    [ "$(sysctl -n hw.optional.arm64 2>/dev/null || printf 0)" = 1 ] || fail macos_intel_unsupported
    [ "$(sysctl -in sysctl.proc_translated 2>/dev/null || printf 0)" = 1 ] || fail macos_intel_unsupported
    platform=darwin_arm64;;
  Linux:x86_64) platform=linux_x64;;
  Linux:aarch64|Linux:arm64) platform=linux_arm64;;
  *) fail unsupported_platform;;
esac
if [ "$os" = Linux ]; then
  libc=$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
  case "$libc" in glibc\ *) ;; *) fail linux_glibc_required;; esac
fi

manifest_url=$(json_string manifest_url); manifest_bytes=$(json_number manifest_bytes); manifest_sha=$(json_string manifest_sha256); validate_sha "$manifest_sha"
component_key=$(json_string "${platform}_component_key"); validate_sha "$component_key"
archive_url=$(json_string "${platform}_archive_url"); archive_bytes=$(json_number "${platform}_archive_bytes"); archive_sha=$(json_string "${platform}_archive_sha256"); validate_sha "$archive_sha"
integrity_relative=$(json_string "${platform}_integrity_path"); integrity_sha=$(json_string "${platform}_integrity_sha256"); validate_sha "$integrity_sha"
file_count=$(json_number "${platform}_file_count"); node_relative=$(json_string "${platform}_node_path"); cli_relative=$(json_string "${platform}_cli_path")
validate_relative "$node_relative"; validate_relative "$cli_relative"
entry=$(json_string app_entry)

home=${HOME:-}
[ -n "$home" ] || fail missing_home
case "$os" in Darwin) cache="$home/Library/Caches/tiangong-lca/runtimes/v1";; *) cache="${XDG_CACHE_HOME:-$home/.cache}/tiangong-lca/runtimes/v1";; esac
marker="$cache/.runtime-cache.json"
if [ -e "$cache" ] && [ -L "$cache" ]; then fail cache_is_link; fi
mkdir -p "$cache"
if [ -e "$marker" ]; then [ -f "$marker" ] && [ ! -L "$marker" ] && [ "$(cat "$marker")" = '{"schema":"tiangong-lca.runtime-cache.v1"}' ] || fail cache_marker_invalid
else
  if [ -e "$marker" ]; then
    [ -f "$marker" ] && [ ! -L "$marker" ] && [ "$(cat "$marker")" = '{"schema":"tiangong-lca.runtime-cache.v1"}' ] || fail cache_marker_invalid
  else
    [ "$(find "$cache" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d '[:space:]')" = 0 ] || { [ -e "$marker" ] || fail cache_not_owned; }
    if ! (set -C; printf '%s\n' '{"schema":"tiangong-lca.runtime-cache.v1"}' > "$marker") 2>/dev/null; then
      tries=0; while [ ! -s "$marker" ] && [ "$tries" -lt 20 ]; do tries=$((tries + 1)); sleep 0.05; done
      [ "$(cat "$marker" 2>/dev/null || true)" = '{"schema":"tiangong-lca.runtime-cache.v1"}' ] || fail cache_marker_invalid
    fi
  fi
fi

lock_dir="$cache/.bootstrap.lock"; owned_lock=0; waited=0
while ! mkdir "$lock_dir" 2>/dev/null; do
  [ ! -L "$lock_dir" ] || fail bootstrap_lock_link
  owner_pid=$(cat "$lock_dir/pid" 2>/dev/null || true); owner_host=$(cat "$lock_dir/host" 2>/dev/null || true); this_host=$(hostname)
  if [ -z "$owner_pid" ] || [ -z "$owner_host" ]; then
    waited=$((waited + 1)); [ "$waited" -le 120 ] || fail bootstrap_lock_invalid
    sleep 0.05; continue
  fi
  case "$owner_pid" in *[!0-9]*) fail bootstrap_lock_invalid;; esac
  if [ "$owner_host" = "$this_host" ] && ! kill -0 "$owner_pid" 2>/dev/null; then rm -f "$lock_dir/pid" "$lock_dir/host"; rmdir "$lock_dir" || fail bootstrap_lock_recovery; continue; fi
  waited=$((waited + 1)); [ "$waited" -le 120 ] || fail bootstrap_lock_timeout; sleep 0.25
done
owned_lock=1; printf '%s\n' "$$" > "$lock_dir/pid"; hostname > "$lock_dir/host"
temp=$(mktemp -d "$cache/.bootstrap-stage.XXXXXX")
cleanup() { rm -rf "$temp"; if [ "$owned_lock" = 1 ]; then rm -f "$lock_dir/pid" "$lock_dir/host"; rmdir "$lock_dir" 2>/dev/null || true; fi; }
trap cleanup EXIT HUP INT TERM

mkdir -p "$cache/manifests" "$cache/components"
manifest_file="$cache/manifests/$manifest_sha.json"
if [ -e "$manifest_file" ]; then verify_file "$manifest_file" "$manifest_bytes" "$manifest_sha"
else download "$manifest_url" "$temp/manifest" "$manifest_bytes"; verify_file "$temp/manifest" "$manifest_bytes" "$manifest_sha"; mv "$temp/manifest" "$manifest_file"; fi

target="$cache/components/$component_key"; root="$target/root"
if [ -d "$root" ]; then verify_component_root "$root" "$integrity_relative" "$integrity_sha" "$file_count"
else
  [ ! -e "$target" ] || fail incomplete_component
  download "$archive_url" "$temp/component.tar.gz" "$archive_bytes"; verify_file "$temp/component.tar.gz" "$archive_bytes" "$archive_sha"
  mkdir "$temp/root"; checksum_source="$temp/root/$integrity_relative"
  # Read the trusted checksum artifact first without extracting any other entry.
  mkdir -p "$(dirname -- "$checksum_source")"
  tar -xOzf "$temp/component.tar.gz" "$integrity_relative" > "$checksum_source" || fail integrity_extract_failed
  [ "$(hash_file "$checksum_source")" = "$integrity_sha" ] || fail integrity_file_changed
  validate_archive "$temp/component.tar.gz" "$integrity_relative" "$file_count" "$temp/list" "$temp/verbose" "$temp/expected"
  rm -rf "$temp/root"; mkdir "$temp/root"; tar -xzf "$temp/component.tar.gz" -C "$temp/root" || fail archive_extract_failed
  verify_component_root "$temp/root" "$integrity_relative" "$integrity_sha" "$file_count"
  publish="$temp/component"; mkdir "$publish"; mv "$temp/root" "$publish/root"; mv "$publish" "$target"
fi
node="$root/$node_relative"; cli="$root/$cli_relative"
[ -x "$node" ] && [ -f "$cli" ] && [ ! -L "$node" ] && [ ! -L "$cli" ] || fail bootstrap_entry_invalid
cwd=$(pwd -P)
trap - EXIT HUP INT TERM; cleanup
exec env -i HOME="$home" USERPROFILE="${USERPROFILE:-$home}" PATH="/usr/bin:/bin" TMPDIR="${TMPDIR:-/tmp}" TEMP="${TEMP:-/tmp}" TMP="${TMP:-/tmp}" LANG="${LANG:-C}" LC_ALL="${LC_ALL:-C}" TZ="${TZ:-UTC}" TIANGONG_LCA_API_BASE_URL="${TIANGONG_LCA_API_BASE_URL:-}" TIANGONG_LCA_SUPABASE_PUBLISHABLE_KEY="${TIANGONG_LCA_SUPABASE_PUBLISHABLE_KEY:-}" TIANGONG_LCA_OAUTH_CLIENT_ID="${TIANGONG_LCA_OAUTH_CLIENT_ID:-}" TIANGONG_LCA_OAUTH_REDIRECT_URI="${TIANGONG_LCA_OAUTH_REDIRECT_URI:-}" TIANGONG_LCA_REGION="${TIANGONG_LCA_REGION:-}" TIANGONG_LCA_AUTH_MODE="${TIANGONG_LCA_AUTH_MODE:-}" TIANGONG_LCA_SESSION_FILE="${TIANGONG_LCA_SESSION_FILE:-}" TIANGONG_LCA_DISABLE_SESSION_CACHE="${TIANGONG_LCA_DISABLE_SESSION_CACHE:-}" TIANGONG_LCA_FORCE_REAUTH="${TIANGONG_LCA_FORCE_REAUTH:-}" TIANGONG_LCA_ACCESS_TOKEN="${TIANGONG_LCA_ACCESS_TOKEN:-}" "$node" "$cli" runtime exec --manifest "$manifest_file" --manifest-sha256 "$manifest_sha" --cache-dir "$cache" --entry "$entry" --cwd "$cwd" -- "$@"
