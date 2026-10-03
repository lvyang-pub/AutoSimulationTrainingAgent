"""CurveAnalyst — 基于 DeepSeek Vision 的训练曲线图像分析 Agent

接收一张或多张训练曲线图（PNG/JPG）+ 自然语言问题，
返回 LLM 对图像的文字分析结论。

可独立运行，也可由 schedular 在每次训练 trial 结束后调用。

用法（直接运行）：
    python Agents/CurveAnalyst/analyst.py \\
        --image path/to/curve.png \\
        --question "整体趋势是什么样子的？收敛了吗？"

用法（Python 调用）：
    from Agents.CurveAnalyst.analyst import CurveAnalyst
    agent = CurveAnalyst()
    result = agent.analyze("path/to/curve.png", "整体趋势是什么样子的？")
    print(result.answer)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass

from openai import OpenAI

# DeepSeek Vision 使用 deepseek-chat 模型（支持图像输入）
VISION_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 默认系统提示：定位为 RL 训练分析专家
DEFAULT_SYSTEM = (
    "你是一位强化学习训练分析专家。"
    "用户会提供训练曲线图和分析问题，你需要仔细观察图像，"
    "给出简洁、准确、有指导意义的分析结论。"
    "聚焦于：趋势方向、收敛情况、异常波动、以及可能的改进方向。"
    "回答用中文，控制在 200 字以内，除非用户要求详细分析。"
)


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        # 尝试从 agent 所在目录向上找 .env
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(4):
            candidate = os.path.join(here, ".env")
            if os.path.exists(candidate):
                path = candidate
                break
            here = os.path.dirname(here)
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


def _get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未设置。请在 .env 文件或环境变量中配置。"
        )
    return key


def _image_to_base64(image_path: str) -> tuple[str, str]:
    """将本地图片转为 base64 data URL，返回 (data_url, mime_type)。"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(ext, "image/png")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}", mime


@dataclass
class AnalysisResult:
    question: str
    answer: str
    image_paths: list[str]
    model: str
    prompt_tokens: int
    completion_tokens: int

    def to_dict(self) -> dict:
        return self.__dict__

    def __str__(self) -> str:
        return self.answer


class CurveAnalyst:
    """DeepSeek Vision 图像分析 Agent。"""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM,
                 api_key: str | None = None,
                 model: str = VISION_MODEL,
                 temperature: float = 0.2):
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or _get_api_key(),
            base_url=DEEPSEEK_BASE_URL,
        )

    def analyze(self, image_paths: str | list[str], question: str) -> AnalysisResult:
        """分析一张或多张图片，回答 question。

        image_paths: 本地文件路径（或路径列表）；支持 PNG/JPG/GIF/WebP。
        question:    自然语言问题，例如"整体趋势是什么样子的？"
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # 构建 content：先放所有图片，再放问题文字
        content = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"图片文件不存在: {path}")
            data_url, _ = _image_to_base64(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "auto"},
            })

        content.append({"type": "text", "text": question})

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1024,
        )

        answer = resp.choices[0].message.content or ""
        usage = resp.usage

        return AnalysisResult(
            question=question,
            answer=answer,
            image_paths=image_paths,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def analyze_trial(self, trial_dir: str, question: str) -> AnalysisResult:
        """便捷方法：直接传入 trial 文件夹，自动找 curve.png。"""
        curve_path = os.path.join(trial_dir, "curve.png")
        if not os.path.exists(curve_path):
            raise FileNotFoundError(
                f"未找到训练曲线图: {curve_path}\n"
                "请确保训练时开启了可视化（visualize=True）。"
            )
        return self.analyze(curve_path, question)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CurveAnalyst: 用 DeepSeek Vision 分析训练曲线图")
    ap.add_argument("--image", required=True, nargs="+",
                    help="图片路径（可多张）")
    ap.add_argument("--question", "-q", required=True,
                    help="分析问题，例如：整体趋势是什么样子的？")
    ap.add_argument("--output", "-o", help="将结果保存为 JSON 文件（可选）")
    args = ap.parse_args()

    agent = CurveAnalyst()
    result = agent.analyze(args.image, args.question)

    print(f"\n问题: {result.question}")
    print(f"\n分析结论:\n{result.answer}")
    print(f"\n(tokens: prompt={result.prompt_tokens}, "
          f"completion={result.completion_tokens})")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
