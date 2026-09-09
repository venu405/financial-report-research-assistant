#!/usr/bin/env bash
# P3 验证脚本：多知识库隔离 + CRUD + 更新
# 用法：bash verify_p3.sh   （需后端已启动在 :8000）
# 所有 curl 带 --noproxy "*" 规避代理干扰（交接文档踩坑①）
BASE=http://127.0.0.1:8000
CURL="curl -s --noproxy 127.0.0.1,localhost"

echo "=== 1. 列知识库（迁移后历史数据应归入 default）==="
$CURL "$BASE/kb/kbs"; echo

echo "=== 2. default 库文档列表（历史 2 份应可见）==="
$CURL "$BASE/kb/docs?kb_id=default"; echo

# 创建一份测试文档上传到 product 库
TMP="$(mktemp "${TMPDIR:-/tmp}/p3_test.XXXXXX.md")"
trap 'rm -f "$TMP"' EXIT
printf "# 产品手册\n本产品保修期为 2 年。\n电池容量 5000mAh。\n充电功率 65W。\n" > "$TMP"

echo "=== 3. 上传到 product 库（应返回 doc_id + kb_id=product）==="
$CURL -X POST "$BASE/kb/ingest" -F "file=@$TMP" -F "title=产品手册" -F "kb_id=product"; echo

echo "=== 4. 隔离验证：问 product 库『保修期多久』（应答到 2 年）==="
$CURL -X POST "$BASE/kb/ask" -H "Content-Type: application/json" \
  -d '{"question":"保修期多久","kb_id":"product"}'; echo

echo "=== 5. 隔离验证：问 product 库『超过多少万元必须招投标』（default 内容，应查不到）==="
$CURL -X POST "$BASE/kb/ask" -H "Content-Type: application/json" \
  -d '{"question":"超过多少万元必须招投标","kb_id":"product"}'; echo

echo "=== 6. default 库问答正常（应答到 5 万元）==="
$CURL -X POST "$BASE/kb/ask" -H "Content-Type: application/json" \
  -d '{"question":"超过多少万元必须招投标","kb_id":"default"}'; echo

echo "=== 7. 列知识库（应含 default + product）==="
$CURL "$BASE/kb/kbs"; echo

echo "=== 8. 列 product 库文档 ==="
$CURL "$BASE/kb/docs?kb_id=product"; echo

# 取 product 库第一个 doc_id 做 PUT 更新验证
DOC_ID=$($CURL "$BASE/kb/docs?kb_id=product" | grep -o '"doc_id":"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -n "$DOC_ID" ]; then
  printf "# 产品手册（修订版）\n保修期延长至 3 年。\n" > "$TMP"
  echo "=== 9. PUT 更新文档 $DOC_ID（保持 doc_id，内容更新为 3 年）==="
  $CURL -X PUT "$BASE/kb/docs/$DOC_ID" -F "file=@$TMP" -F "title=产品手册v2" -F "kb_id=product"; echo
  echo "=== 10. 更新后再问保修期（应答到 3 年）==="
  $CURL -X POST "$BASE/kb/ask" -H "Content-Type: application/json" \
    -d '{"question":"保修期多久","kb_id":"product"}'; echo
  echo "=== 11. 删除该文档 ==="
  $CURL -X DELETE "$BASE/kb/docs/$DOC_ID"; echo
else
  echo "=== 9-11 跳过：未取到 product 库 doc_id ==="
fi

echo "=== 验证结束 ==="
