import argparse, os, json, re
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import sacrebleu
from rouge_chinese import Rouge
import jieba

POSSIBLE_REPORT_KEYS = ["report", "medical_report", "mrg_report", "summary"]

ALIAS = {
    "主诉": ["主诉", "就诊原因"],
    "现病史": ["现病史", "病史", "近期情况"],
    "查体要点": ["查体要点", "体格检查", "查体", "阳性体征"],
    "辅助检查": ["辅助检查", "检查", "化验", "实验室检查", "影像学检查", "检验"],
    "既往史": ["既往史", "既往疾病史", "过敏史", "家族史", "个人史"],
    "诊断": ["诊断", "初步诊断", "考虑"],
    "建议": ["建议", "治疗建议", "处置", "医嘱", "随访建议"],
}
SECS = ["主诉","现病史","辅助检查","既往史","诊断","建议"]


def load_jsonl(path: str):
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    obj = json.load(open(path, "r", encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return list(obj.values())
    raise ValueError("无法识别的JSON结构：期望列表或字典")


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


def normalize_report_to_canon(raw_report, secs):
    from collections import defaultdict
    out = defaultdict(list)
    items = []
    if isinstance(raw_report, list):
        items = [x for x in raw_report if isinstance(x, dict)]
    elif isinstance(raw_report, dict):
        items = [raw_report]
    elif isinstance(raw_report, str):
        return {k: (raw_report if k == "现病史" else "无") for k in secs}

    def map_key(k: str):
        k = str(k).strip()
        for canon, aliases in ALIAS.items():
            if k == canon or k in aliases:
                return canon
        return None

    for d in items:
        for k, v in d.items():
            canon = map_key(k)
            if canon in secs:
                v = str(v).strip()
                if v: out[canon].append(v)

    return {k: ("；".join(out[k]) if out[k] else "无") for k in secs}

def refs_to_bodies_list(ref_raw, secs):
    refs = ref_raw if isinstance(ref_raw, list) else [ref_raw]
    bodies = []
    for r in refs:
        r_can = normalize_report_to_canon(r, secs=secs)
        bodies.append("\n".join([r_can[k] for k in secs]))
    return bodies

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

def normalize_item(raw):
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
        "请根据以下医患对话生成规范的门诊病历/报告，严格分为6节：主诉、现病史、查体要点、辅助检查、既往史、建议。\n"
        "每节一句话高度概括；要求：每节尽量保留时间/部位/程度/持续时间/用药名/检查项目及数值；否认写‘否认××’，无信息写‘无’；"
        "不要客套话。"
    )

    return {"instruction": instr, "input": text, "output": ref}

def build_messages(instruction: str, input_text: str):
    system = (
        "你是严谨的中文医学助手。输出严格分为6节：主诉、现病史、查体要点、辅助检查、既往史、建议。"
        "每节一句话高度概括；出现的时间/部位/程度/持续时间/用药名/检查项目及数值尽量逐一保留；"
        "明确否认的信息写‘否认××’；未知写‘无’；不得输出与这6节无关的任何内容。"
    )
    user = (
        "请将对话内容整理为门诊病历/报告，使用以下标题并按顺序输出：\n"
        "主诉：\n现病史：\n辅助检查：\n既往史：\n建议："
    )
    if input_text and input_text.strip():
        user += "\n\n【对话全文】\n" + input_text.strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def postprocess_pred(text: str, secs=SECS):
    t = re.sub(r"\*+", "", text)                       # 去 Markdown 粗体
    t = re.sub(r"^\s*[-•·]+\s*", "", t, flags=re.M)    # 去项目符
    t = t.replace(":", "：")
    sec = sec_cut(t)
    for k in secs:
        val = (sec.get(k) or "").strip()
        if not val:
            sec[k] = "无"
        else:
            lines = [x.strip("；。 \t") for x in val.splitlines() if x.strip()]
            sec[k] = "；".join(lines) if lines else "无"
    return "\n".join(sec[k] for k in secs)

def refs_to_bodies_list(ref_raw, secs=SECS):
    refs = ref_raw if isinstance(ref_raw, list) else [ref_raw]
    bodies = []
    for r in refs:
        r_can = normalize_report_to_canon(r, secs=secs)  # 与参考端同一映射
        bodies.append("\n".join([r_can[k] for k in secs]))
    return bodies


def bleu_zh(refs: List[str], hyps: List[str]) -> float:
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score if hyps else 0.0


_ROUGE = Rouge()  # 计算 F1 by default

def rouge_zh_avg(refs: List[str], hyps: List[str]) -> Tuple[float, float, float]:
    if not hyps:
        return 0.0, 0.0, 0.0
    scores = _ROUGE.get_scores(hyps, refs, avg=True)  # {'rouge-1':{'f':..,'p':..,'r':..}, ...}
    return float(scores['rouge-1']['f']), float(scores['rouge-2']['f']), float(scores['rouge-l']['f'])


def strip_titles(text: str) -> str:
    # 去掉节标题，仅比正文
    if not text:
        return ""
    pat = re.compile(rf"^({'|'.join(map(re.escape, SECS))})：", flags=re.M)
    return pat.sub("", text)


def refs_to_body_text(ref):
    ref6 = normalize_report_to_canon(ref, secs=SECS)
    return "\n".join([ref6[k] for k in SECS])


def eval_promptcblue_style(refs_raw: List[Any], preds: List[str], compare_body_only: bool = True) -> Dict[str, float]:
    refs_txt = [refs_to_body_text(r) for r in refs_raw]
    hyps_txt = [strip_titles(p) if compare_body_only else p for p in preds]
    bleu = bleu_zh(refs_txt, hyps_txt)
    r1, r2, rl = rouge_zh_avg(refs_txt, hyps_txt)
    avg_len = float(np.mean([len(x) for x in hyps_txt])) if hyps_txt else 0.0
    return {
        "BLEU": bleu,
        "ROUGE-1": r1,
        "ROUGE-2": r2,
        "ROUGE-L": rl,
        "avg_chars": avg_len,
        "n_eval": len(hyps_txt),
    }


SEC_TITLES = [f"{k}：" for k in SECS]


def sec_cut(text: str) -> Dict[str, str]:
    res = {k: "" for k in SECS}
    if not text:
        return res
    for i, k in enumerate(SECS):
        nxt = SECS[i + 1] if i < len(SECS) - 1 else None
        pat = re.compile(re.escape(k) + "：" + (r"(.+?)" + ("(?=" + re.escape(nxt) + "：)" if nxt else "$")), re.S)
        m = re.search(pat, text)
        res[k] = (m.group(1).strip() if m else "")
    return res


def rougeL_char(a: str, b: str) -> float:
    return Rouge().get_scores([a], [b])[0]['rouge-l']['f']


def eval_by_sections(refs_raw: List[Any], preds: List[str]) -> Dict[str, float]:
    scores = {k: [] for k in SECS}
    for r, p in zip(refs_raw, preds):
        ref7 = normalize_report_to_canon(r)
        pred_sec = sec_cut(p)
        for k in SECS:
            scores[k].append(rougeL_char(ref7[k], pred_sec[k]))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}

def eval_promptcblue_style_multiref(refs_raw, preds, secs, compare_body_only=True):
    rouge = Rouge()
    r1s, r2s, rls, bleus = [], [], [], []

    for ref_raw, hyp in zip(refs_raw, preds):
        hyp_txt = postprocess_pred(hyp) if compare_body_only else hyp
        ref_texts = refs_to_bodies_list(ref_raw, secs=secs)  # list[str]
        scores = [rouge.get_scores(hyp_txt, r, avg=True) for r in ref_texts]
        r1s.append(max(s['rouge-1']['f'] for s in scores))
        r2s.append(max(s['rouge-2']['f'] for s in scores))
        rls.append(max(s['rouge-l']['f'] for s in scores))

        bleus.append(sacrebleu.sentence_bleu(hyp_txt, ref_texts, tokenize="zh").score)

    return {
        "BLEU": float(np.mean(bleus)) if bleus else 0.0,
        "ROUGE-1": float(np.mean(r1s)) if r1s else 0.0,
        "ROUGE-2": float(np.mean(r2s)) if r2s else 0.0,
        "ROUGE-L": float(np.mean(rls)) if rls else 0.0,
        "avg_chars": float(np.mean([len(strip_titles(p)) if compare_body_only else len(p) for p in preds])) if preds else 0.0,
        "n_eval": len(preds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default='IMCS-V2-MRG/IMCS-V2_dev.json', help="JSON/JSONL 路径")
    ap.add_argument("--models", nargs="+", default=["qwen2.5-0.5b=/raid/lhk/qwen2.5-0.5b","qwen2.5-1.5b=/raid/lhk/qwen2.5-1.5b","qwen2.5-3b=/raid/lhk/qwen2.5-3b","qwen3-4b=/raid/lhk/qwen3-4b","qwen2.5-7b=/raid/lhk/qwen2.5-7b"], help="别名=模型路径")
    ap.add_argument("--save_dir", type=str, default='result')
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16","float16","half","auto"])
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_util", type=float, default=0.91)

    # 评测风格
    ap.add_argument("--eval_mode", type=str, default="promptcblue", choices=["promptcblue", "naive"], help="promptcblue 使用 sacrebleu(tokenize=zh)+rouge-chinese")
    ap.add_argument("--compare_body_only", action="store_true", help="预测去掉标题后再评测")
    ap.add_argument("--dump_sections", action="store_true", help="打印分节ROUGE用于诊断")

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    data_raw = load_jsonl(args.data)
    data = [normalize_item(x) for x in data_raw]

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
            for i, (ex, out) in enumerate(zip(data, outputs)):
                text = out.outputs[0].text if out.outputs else ""
                rec = {"instruction": ex["instruction"], "input": ex["input"], "ref": ex["output"], "pred": text}
                fpred.write(json.dumps(rec, ensure_ascii=False) + "\n")
                flog.write(
                    f"### Sample {i}\n[INSTRUCTION]\n{ex['instruction']}\n[INPUT]\n{ex['input']}\n[PRED]\n{text}\n[REF]\n{ex['output']}\n\n"
                )
                preds.append(text)
                refs_raw.append(ex["output"])


        m = eval_promptcblue_style_multiref(refs_raw, preds,secs=SECS, compare_body_only=args.compare_body_only)
        if args.dump_sections:
            sec_scores = eval_by_sections(refs_raw, preds)
            # 追加到 CSV 列
            for k, v in sec_scores.items():
                m[f"secRL_{k}"] = v

        m["model"] = alias
        rows.append(m)
        
        try:
            llm.shutdown()
        except Exception:
            pass
        del llm

        # 进一步收尾
        del tokenizer
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 汇总
    df = pd.DataFrame(rows)
    # 为了和常见做法一致，按 ROUGE-L 降序
    df = df.sort_values("ROUGE-L", ascending=False)
    out_csv = os.path.join(args.save_dir, "metrics_promptcblue.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved metrics to:", out_csv)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
