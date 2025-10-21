import json
import argparse
import csv
import sys
from typing import List
from rouge import Rouge
from pathlib import Path
from tqdm import tqdm
import sacrebleu

def bleu_zh(refs: List[str], hyps: List[str]) -> float:
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score if hyps else 0.0

def compute_avg_chars(texts: List[str]) -> float:
    """计算预测文本的平均字符数"""
    if not texts:
        return 0.0
    return round(sum(len(t) for t in texts) / len(texts), 1)

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def process(title, delimiter=''):
    x = []
    for key, value in title.items():
        x.append(key + '：' + value)
    return delimiter.join(x)


def compute_bleu(source, targets):
    """计算单个生成文本与多个参考文本的BLEU分数"""
    try:
        source_tokens = source.split()
        target_tokens_list = [target.split() for target in targets]
        bleu_score = bleu_zh(source, targets)
        return round(bleu_score, 4)
    except Exception as e:
        print(f"BLEU计算出错：{e}")
        return 0.0


def compute_metrics(source, targets):
    """同时计算Rouge和BLEU指标"""
    try:
        # 计算Rouge分数
        r1, r2, rl = 0, 0, 0
        n = len(targets)
        for target in targets:
            source_str, target_str = ' '.join(source), ' '.join(target)
            scores = Rouge().get_scores(hyps=source_str, refs=target_str)
            r1 += scores[0]['rouge-1']['f']
            r2 += scores[0]['rouge-2']['f']
            rl += scores[0]['rouge-l']['f']
        rouge_scores = {
            'rouge-1': r1 / n,
            'rouge-2': r2 / n,
            'rouge-l': rl / n,
            'avg_rouge': (r1 + r2 + rl) / (3 * n)
        }
        
        # 计算BLEU分数
        source_str = ' '.join(source)
        target_strs = [' '.join(target) for target in targets]
        bleu_score = compute_bleu(source_str, target_strs)
        return {** rouge_scores, 'bleu': bleu_score}
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
            'avg_rouge': 0.0,
            'bleu': 0.0
        }


def compute_all_metrics(sources, targets):
    """计算所有样本的平均Rouge和BLEU指标"""
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
        'bleu': 0.0
    }
    for source, target in zip(sources, targets):
        metrics = compute_metrics(source, target)
        scores['rouge-1'] += metrics['rouge-1']
        scores['rouge-2'] += metrics['rouge-2']
        scores['rouge-l'] += metrics['rouge-l']
        scores['bleu'] += metrics['bleu']
    
    total = len(targets)
    return {
        'rouge-1': round(scores['rouge-1'] / total, 4),
        'rouge-2': round(scores['rouge-2'] / total, 4),
        'rouge-l': round(scores['rouge-l'] / total, 4),
        'avg_rouge': round((scores['rouge-1'] + scores['rouge-2'] + scores['rouge-l']) / (3 * total), 4),
        'bleu': round(scores['bleu'] / total, 4)
    }


def evaluate_single(gold_data, pred_path):
    """评估单个预测文件（含样本处理信息打印）"""
    
    pred_path = 'result/'+pred_path
    pred_filename = Path(pred_path).name
    total_samples = len(gold_data)  # 总样本数
    
    try:
        # 加载预测文件
        print(f"\n正在加载预测文件：{pred_path}")
        pred_data = load_json(pred_path)
        print(f"成功加载 {pred_filename}（共需处理 {total_samples} 个样本）")
        
        # 处理样本（添加样本信息打印）
        golds, preds = [], []
        # 使用tqdm显示进度条，同时用enumerate获取样本索引
        for idx, (pid, sample) in enumerate(tqdm(
            gold_data.items(),
            desc=f"处理 {pred_filename}",
            total=total_samples,
            leave=True
        )):
            # 每100个样本打印一次详细信息（避免输出过多）
            if (idx + 1) % 100 == 0 or (idx + 1) == total_samples:
                print(f"   已处理样本 {idx+1}/{total_samples}，当前样本ID：{pid}")
            
            # 提取参考报告
            title1, title2 = sample['report'][0], sample['report'][1]
            golds.append([process(title1), process(title2)])
            
            # 检查样本ID是否存在
            if pid not in pred_data:
                raise ValueError(f"缺失样本ID：{pid}")
            preds.append(process(pred_data[pid]))
        
        # 计算指标
        print(f"{pred_filename} 样本处理完成，正在计算指标...")
        metrics = compute_all_metrics(preds, golds)
        avg_chars = compute_avg_chars(preds)
        return {**metrics, 'avg_chars': avg_chars}
    
    except Exception as e:
        print(f"{pred_filename} 处理失败：{str(e)}")
        return f"处理失败：{str(e)}"


def save_to_csv(results, output_path="result.csv"):
    """保存包含Rouge和BLEU的结果到CSV"""
    fieldnames = [
        'pred_file', 'rouge-1', 'rouge-2', 'rouge-l', 'avg_rouge', 'bleu', 'avg_chars', 'status'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in results:
            row = {
                'pred_file': item['pred_file'],
                'rouge-1': item.get('rouge-1', '-'),
                'rouge-2': item.get('rouge-2', '-'),
                'rouge-l': item.get('rouge-l', '-'),
                'avg_rouge': item.get('avg_rouge', '-'),
                'bleu': item.get('bleu', '-'),
                'avg_chars': item.get('avg_chars', '-')
            }
            writer.writerow(row)
    
    print(f"\n所有比对结果已保存至CSV文件：{output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', type=str, default="/raid/lhk/homework/medbench-llm-inference/IMCS-V2-MRG/test.json", help='参考报告文件路径（唯一基准）')
    parser.add_argument('--pred_paths', type=str, nargs='+', 
                        default=["qwen2.5-0.5b_preds.json", "qwen2.5-1.5b_preds.json", "qwen2.5-3b_preds.json", "qwen3-4b_preds.json", "qwen2.5-7b_preds.json"],
                        help='多个预测报告文件路径（按空格分隔，将按传入顺序处理）')
    parser.add_argument('--output_csv', type=str, default='result/metrics_promptcblue.csv', help='CSV结果保存路径')
    
    args = parser.parse_args()

    # 初始化信息
    print("="*80)
    print("多数据集比对评估工具（含Rouge+BLEU，按顺序处理）")
    print("="*80)
    
    # 加载参考报告
    try:
        print(f"\n正在加载参考报告：{args.gold_path}")
        gold_data = load_json(args.gold_path)
        total_gold_samples = len(gold_data)
        print(f"参考报告加载成功！共包含 {total_gold_samples} 个样本")
    except Exception as e:
        print(f"参考报告加载失败：{str(e)}")
        exit(1)
    
    # 按传入顺序处理每个预测文件
    comparison_results = []
    total_pred_files = len(args.pred_paths)
    print(f"\n共检测到 {total_pred_files} 个预测文件，将按以下顺序处理：")
    for idx, path in enumerate(args.pred_paths, 1):
        print(f"   {idx}. {Path(path).name}")
    print("-"*50)
    
    # 遍历处理
    for file_idx, pred_path in enumerate(args.pred_paths, 1):
        pred_filename = Path(pred_path).name
        print(f"\n开始处理第 {file_idx}/{total_pred_files} 个文件：{pred_filename}")
        print("-"*50)
        
        # 评估单个文件
        result = evaluate_single(gold_data, pred_path)
        
        # 保存结果到汇总列表
        if isinstance(result, dict):
            print(f"{pred_filename} 评估完成！指标如下：")
            print(f"   Rouge-1: {result['rouge-1']} | Rouge-2: {result['rouge-2']} | Rouge-L: {result['rouge-l']}")
            print(f"   平均Rouge: {result['avg_rouge']} | BLEU: {result['bleu']}")
            comparison_results.append({
                'pred_file': pred_filename,** result
            })
        else:
            comparison_results.append({
                'pred_file': pred_filename,
                'error': result
            })
        print("-"*50)
    
    # 打印最终汇总表格
    print("\n" + "="*80)
    print("所有文件处理完成！多数据集比对结果汇总（含Rouge和BLEU）")
    print("="*80)
    print(f"{'预测文件':<20} | {'rouge-1':<8} | {'rouge-2':<8} | {'rouge-l':<8} | {'平均rouge':<8} | {'bleu':<8} | 状态")
    print("-"*80)
    for item in comparison_results:
        if 'error' in item:
            print(f"{item['pred_file']:<20} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {item['error']}")
        else:
            print(f"{item['pred_file']:<20} | {item['rouge-1']:<8} | {item['rouge-2']:<8} | {item['rouge-l']:<8} | {item['avg_rouge']:<8} | {item['bleu']:<8} | 正常")
    print("="*80)
    
    # 保存结果到CSV
    save_to_csv(comparison_results, args.output_csv)