import argparse, os, json, re
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import sacrebleu
from rouge import Rouge
import jieba

POSSIBLE_REPORT_KEYS = ["report", "medical_report", "mrg_report", "summary"]

ALIAS = {
    "主诉": ["主诉", "就诊原因"],
    "现病史": ["现病史", "病史", "近期情况"],
    "辅助检查": ["辅助检查", "检查", "化验", "实验室检查", "影像学检查", "检验"],
    "既往史": ["既往史", "既往疾病史", "过敏史", "家族史", "个人史"],
    "诊断": ["诊断", "初步诊断", "考虑"],
    "建议": ["建议", "治疗建议", "处置", "医嘱", "随访建议"],
}

SECS = ["主诉","现病史","辅助检查","既往史","诊断","建议"]

from collections import OrderedDict
from typing import Any, Mapping, Iterable

def load_records(path: str,
                 id_key: str = "id",
                 alt_id_keys: Iterable[str] = ("rid", "record_id", "编号"),
                 drop_id_from_item: bool = True) -> "OrderedDict[str, Any]":
    alt_keys = (id_key,) + tuple(alt_id_keys)

    def _coerce_id(v: Any) -> str:
        if v is None:
            raise ValueError("编号为空")
        return str(v)

    def _pop_id(d: Mapping[str, Any]) -> tuple[str, dict]:
        rid = None
        hit = None
        for k in alt_keys:
            if k in d:
                rid = d[k]; hit = k; break
        if rid is None:
            raise ValueError(f"缺少编号字段（可选字段：{', '.join(alt_keys)}）")
        item = dict(d)
        if drop_id_from_item and hit in item:
            item.pop(hit, None)
        return _coerce_id(rid), item

    od = OrderedDict()

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"第{ln}行不是对象：{type(obj)}")

                # 形如 {"10035922": {...}}
                if len(obj) == 1 and all(isinstance(k, (str, int)) for k in obj.keys()):
                    rid, item = next(iter(obj.items()))
                    rid = _coerce_id(rid)
                    if not isinstance(item, dict):
                        raise ValueError(f"第{ln}行的记录不是对象：{type(item)}")
                else:
                    rid, item = _pop_id(obj)

                if rid in od:
                    raise ValueError(f"重复编号：{rid}（第{ln}行）")
                od[rid] = item
        return od

    # 非 jsonl，当作 .json
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        # 顶层就是 {编号: 记录}
        for k, v in obj.items():
            rid = _coerce_id(k)
            if not isinstance(v, dict):
                raise ValueError(f"编号 {rid} 的记录不是对象：{type(v)}")
            if rid in od:
                raise ValueError(f"重复编号：{rid}")
            od[rid] = v
        return od

    if isinstance(obj, list):
        # 顶层是列表：从元素里拿 id_key/alt_id_keys
        for i, row in enumerate(obj, 1):
            if not isinstance(row, dict):
                raise ValueError(f"第{i}个元素不是对象：{type(row)}")
            rid, item = _pop_id(row)
            if rid in od:
                raise ValueError(f"重复编号：{rid}（第{i}个元素）")
            od[rid] = item
        return od

    raise ValueError("无法识别的JSON结构：期望列表或字典")

def canonicalize_keys(d: dict) -> dict:
    out = {k:"无" for k in SECS}
    for canon, aliases in ALIAS.items():
        for k, v in (d or {}).items():
            if str(k).strip() in aliases:
                out[canon] = _clean_text_val(v); break
    return out

def dedupe_suggestions(s: str, max_items=8) -> str:
    if not s or s=="无": return s or "无"
    # 按分号/换行/编号分割
    items = re.split(r"(?:\s*[；;。\n]+|\s*\d+[\.\)）]\s*)", s)
    seen, cleaned = set(), []
    for t in items:
        t = t.strip(" ；;。，.\n")
        if not t: continue
        if t in seen: continue
        seen.add(t)
        cleaned.append(t)
        if len(cleaned) >= max_items: break
    return "；".join(cleaned) if cleaned else "无"

HD = "|".join(SECS)

def normalize_headings(text: str) -> str:
    s = text or ""
    # 1) 任何位置出现【栏名】或[栏名]/(栏名)，一律前面加换行
    s = re.sub(rf"\s*(?=[【\[\(]\s*(?:{HD})\s*[】\]\)])", "\n", s)
    # 2) 把各种括号标题统一成 “栏名：”
    s = re.sub(rf"[【\[\(]\s*({HD})\s*[】\]\)]\s*[:：；]?", r"\1：", s)
    # 3) 全角/半角冒号统一
    s = re.sub(r"(:|：)\s*", "：", s)
    return s

PAT = rf"(?:^|\n)\s*(?:({HD}))\s*[:：]?\s*(.*?)(?=(?:^|\n)\s*(?:{HD})\s*[:：]?|\Z)"

def split_sections(text: str) -> dict:
    txt = normalize_headings(text)
    pairs = re.findall(PAT, txt, flags=re.S|re.M)
    out = {}
    for k, v in pairs:
        out[k] = (v or "").strip()
    return out

def parse_to_six_sections(text: str) -> dict:
    # 1) 尝试 JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return canonicalize_keys({k:_clean_text_val(v) for k,v in obj.items()})
    except Exception:
        pass
    # 2) 标题切分
    kv = split_sections(text)
    d = canonicalize_keys(kv)
    # 3) 单栏清洗
    for k in SECS:
        d[k] = _clean_text_val(d.get(k,"无"))
    # 4) 建议去重（可选）
    d["建议"] = dedupe_suggestions(d.get("建议","无"))
    return d

def _join_dialogue(dlg):
    parts = []
    if isinstance(dlg, list):
        for turn in dlg:
            spk = str(turn.get("speaker") or turn.get("role") or "").strip()
            utt = str(turn.get("sentence") or turn.get("content") or turn.get("text") or "").strip()
            if not utt:
                continue
            parts.append(f"{spk}：{utt}" if spk else utt)
    return "\n".join(parts)

_PLACEHOLDER_RE = re.compile(r"^(暂缺|不详|未知|不明|未详|未述|未记录|未提及|未询|暂无|无资料)(信息|情况|记录|史)?$")

def _clean_text_val(s: str) -> str:
    """对单段文本做清洗：去段末顿号；占位词 -> 无；多行/多分句逐段处理"""
    t = str(s or "").strip()
    if not t:
        return "无"

    # 先把英文冒号替换为中文，统一符号（可选）
    t = t.replace(":", "：")

    # 仅去掉“行末/分句末”的全角句号，不动句中“。”（避免影响缩写等）
    # 注意先按常见分隔切段，再对每段末尾处理
    segments = re.split(r"[；;、,\n]+", t)
    cleaned = []
    for seg in segments:
        seg = seg.strip()
        # 去掉末尾的“。”（如果有）
        seg = re.sub(r"。+$", "", seg)

        if not seg or _PLACEHOLDER_RE.fullmatch(seg):
            cleaned.append("无")
        else:
            cleaned.append(seg)

    # 去空并用中文分号合并；如果全空则给“无”
    cleaned = [x for x in cleaned if x]
    return "；".join(cleaned) if cleaned else "无"

def _clean_ref_structure(ref):
    """递归清洗 ref（支持 str / dict / list[dict|str]）"""
    if isinstance(ref, str):
        return _clean_text_val(ref)
    if isinstance(ref, dict):
        return {k: _clean_text_val(v) for k, v in ref.items()}
    if isinstance(ref, list):
        return [_clean_ref_structure(x) for x in ref]
    return ref  # 其他类型原样返回

def normalize_item(example_id, raw):
    text = _join_dialogue(raw.get("dialogue") or raw.get("conversation") or raw.get("dialog") or [])
    if not text:
        text = raw.get("input") or raw.get("question") or raw.get("prompt") or raw.get("src") or raw.get("context") or ""

    ref = ""
    for k in POSSIBLE_REPORT_KEYS:
        if k in raw and raw[k]:
            ref = raw[k]
            break
    if not ref:
        ref = raw.get("output") or raw.get("answer") or raw.get("target") or ""

    ref = _clean_ref_structure(ref)
    
    instr = raw.get("instruction") or (
        "请根据以下医患对话生成规范的门诊病历/报告，严格分为6节：主诉、现病史、辅助检查、既往史、诊断、建议。\n"
        "每节一句话高度概括；要求尽量保留时间/部位/程度/持续时间/用药名/检查项目及数值；明确否认写‘否认××’，无信息写‘无’；"
        "不得输出与这6节无关的任何内容。"
    )

    return {"id":example_id, "instruction": instr, "input": text, "output": ref}

def build_messages(instruction: str, input_text: str):
    system = (
        "你是严谨的中文医学助手。输出严格分为6节：主诉、现病史、辅助检查、既往史、诊断、建议。"
        "每节一句话高度概括；出现的时间/部位/程度/持续时间/用药名/检查项目及数值尽量逐一保留；"
        "明确否认的信息写‘否认××’；未知写‘无’；不得输出与这6节无关的任何内容。"
    )
    user = (
        "请将对话内容整理为门诊病历/报告，使用以下标题并按顺序输出：\n"
        "主诉：\n现病史：\n辅助检查：\n既往史：\n诊断：\n建议："
    )
    if input_text and input_text.strip():
        user += "\n\n【对话全文】\n" + input_text.strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default='IMCS-V2-MRG/IMCS-V2_test.json', help="JSON/JSONL 路径")
    ap.add_argument("--models", nargs="+", default=["qwen2.5-0.5b=/raid/lhk/qwen2.5-0.5b","qwen2.5-1.5b=/raid/lhk/qwen2.5-1.5b","qwen2.5-3b=/raid/lhk/qwen2.5-3b","qwen3-4b=/raid/lhk/qwen3-4b","qwen2.5-7b=/raid/lhk/qwen2.5-7b"], help="别名=模型路径")
    ap.add_argument("--save_dir", type=str, default='result')
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16","float16","half","auto"])
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_util", type=float, default=0.91)

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    data_raw = load_records(args.data)
    data = [normalize_item(example_id, content) for example_id, content in data_raw.items()]
    # 构造所有消息
    all_msgs: List[List[Dict[str, str]]] = [build_messages(ex["instruction"], ex["input"]) for ex in data]

    rows = []
    for spec in args.models:
        if "=" not in spec:
            raise ValueError(f"Bad --models item: {spec}, expect alias=path_or_repo")
        alias, model_id = spec.split("=", 1)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True, local_files_only=True)

        # 估计最长输入长度
        all_lengths = []
        prompts = []
        for msg in tqdm(all_msgs, desc=f"[{alias}] 构建与估算长度"):
            input_text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            tok = tokenizer(input_text, return_tensors="pt")
            all_lengths.append(int(tok["input_ids"].shape[1]))
            prompts.append(input_text)
        max_input_len = int(max(all_lengths) if all_lengths else 1024) + int(args.max_new_tokens)
        max_input_len = min(max_input_len, 16384)

        # vLLM
        llm = LLM(
            model=model_id,
            trust_remote_code=True,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_util,
            max_model_len=max_input_len,
        )
        sampler = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=min(args.max_new_tokens, 1024),
            stop=None,
        )

        # 生成 & 日志
        log_path = os.path.join(args.save_dir, f"{alias}_log.txt")
        pred_path = os.path.join(args.save_dir, f"{alias}_preds.jsonl")
        outputs = llm.generate(prompts, sampler, use_tqdm=True)

        preds, refs_raw = [], []
        with open(log_path, "w", encoding="utf-8") as flog, open(pred_path, "w", encoding="utf-8") as fpred:
            by_id = {}  # 用于生成 {"10035922": {...}} 的总表

            for i, (ex, out) in enumerate(zip(data, outputs)):
                rid = str(ex.get("id", i))  # 你的 id 在 ex["id"]
                text = out.outputs[0].text if getattr(out, "outputs", None) else ""

                # 把模型输出解析成固定六栏
                pred_struct = parse_to_six_sections(text)

                # 日志：多打印一份结构化JSON便于抽查
                flog.write(
                    "### Sample {idx} (id={rid})\n[INSTRUCTION]\n{ins}\n[INPUT]\n{inp}\n"
                    "[PRED_TEXT]\n{pt}\n[PRED_JSON]\n{pj}\n[REF]\n{rf}\n\n".format(
                        idx=i, rid=rid,
                        ins=ex.get("instruction",""),
                        inp=ex.get("input",""),
                        pt=text,
                        pj=json.dumps(pred_struct, ensure_ascii=False, indent=2),
                        rf=ex.get("output",""),
                    )
                )

                refs_raw.append(ex.get("output",""))

                by_id[rid] = pred_struct
        
        with open(f"result/{alias}_preds.json", "w", encoding="utf-8") as f:
            json.dump(by_id, f, ensure_ascii=False, indent=2)
            
        try:
            llm.shutdown()
        except Exception:
            pass
        del llm

        del tokenizer
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
