#!/usr/bin/env sh
# docker/scripts/tests/test_env_sh.sh
#
# sh docker/scripts/tests/test_env_sh.sh


cat > tmp.env <<'EOF'
# comment
FOO=bar
BAR="baz"
ZED='zed'
EOF
sh -c '. docker/scripts/lib/common.sh; load_env_kv tmp.env; [ "$FOO" = "bar" ] && [ "$BAR" = "baz" ] && [ "$ZED" = "zed" ] && echo OK'
rm -f tmp.env


cat > bad.env <<'EOF'
export X=1
EOF
sh -c '. docker/scripts/lib/common.sh; load_env_kv bad.env' && echo "FAIL" || echo "PASS (expected failure)"
rm -f bad.env
