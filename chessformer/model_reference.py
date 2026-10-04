"""Source-guided NumPy oracle and parameter audit for the pinned trainer subset.

This is NOT TensorFlow execution. It independently translates the old embedding,
Mish/RMSNorm encoder, Shaw terms, policy heads and three value heads. Displacement
and policy maps are executed from verified upstream source. Unsupported branches
(new embedding, smolgen, quantization, extra heads) are not represented.
"""

import argparse
import ast
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import Chessformer, ModelConfig, keras_variance_scaling_
from .reference import check_sources


def source_maps():
    specification = check_sources()
    source = next(e for e in specification["files"] if e["path"] == "tf/tfprocess.py")
    tree = ast.parse(Path(source["local_path"]).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_rpe_map"]
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), source["path"], "exec"), namespace)
    relative = namespace["make_rpe_map"]()
    policy_source = next(e for e in specification["files"] if e["path"] == "tf/attention_policy_map.py")
    tree = ast.parse(Path(policy_source["local_path"]).read_text())
    # Everything before set_zero_sum is the pure NumPy map builder, except imports.
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "set_zero_sum":
            break
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), policy_source["path"], "exec"), namespace)
    return relative, namespace["apm_out"]


def numpy_forward(model, inputs, maps):
    """Float64 NumPy forward using the trainer's documented tensor layouts."""
    cfg = model.cfg
    if cfg.deepnorm != "trainer" or cfg.position != "shaw" or cfg.shaw_cross_term:
        raise ValueError("reference requires trainer DeepNorm and additive Shaw terms")
    relative_map, policy_map = maps
    weights = {name: p.detach().cpu().numpy().astype(np.float64) for name, p in model.named_parameters()}
    x = np.asarray(inputs, dtype=np.float64)

    def dense(x, name):
        value = x @ weights[name + ".weight"].T
        return value + weights[name + ".bias"] if name + ".bias" in weights else value

    def mish(x):
        return x * np.tanh(np.logaddexp(x, 0))

    def rms(x, name):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5) * weights[name + ".weight"]

    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    x = mish(dense(x, "embedding")) * weights["input_scale"] + weights["input_offset"]
    attention = []
    alpha = (2 * cfg.layers) ** -0.25
    h, depth = cfg.heads, cfg.attention_width // cfg.heads
    for layer in range(cfg.layers):
        name = f"blocks.{layer}"
        kernels = np.split(weights[name + ".attn.qkv.weight"].T, 3, axis=-1)
        q, k, v = [(x @ w).reshape(-1, 64, h, depth).transpose(0, 2, 1, 3) for w in kernels]

        def expanded(parameter):
            # Upstream stores [depth*heads,225], then expands as [depth,heads,q,k].
            table = weights[name + ".attn." + parameter].transpose(2, 0, 1).reshape(depth * h, 225)
            return (table @ relative_map).reshape(depth, h, 64, 64)

        # Source rpe_q multiplies Q, corresponding to the additive key table.
        logits = q @ k.swapaxes(-1, -2)
        logits += np.einsum("bhqd,dhqk->bhqk", q, expanded("rk"))
        logits += np.einsum("bhkd,dhqk->bhqk", k, expanded("rq"))
        probs = softmax(logits / np.sqrt(depth))
        attention.append(probs)
        value = probs @ v + np.einsum("bhqk,dhqk->bhqd", probs, expanded("rv"))
        value = value.transpose(0, 2, 1, 3).reshape(-1, 64, cfg.attention_width)
        x = rms(x + dense(value, name + ".attn.proj") * alpha, name + ".norm1")
        x = rms(x + dense(mish(dense(x, name + ".up")), name + ".down") * alpha, name + ".norm2")
    tokens = mish(dense(x, "policy_embedding"))
    output = {"attention": np.stack(attention, axis=1)}
    for name in ("policy", "soft_policy"):
        queries, keys = dense(tokens, name + ".query"), dense(tokens, name + ".key")
        logits = queries @ keys.swapaxes(-1, -2)
        divisor = np.sqrt(keys.shape[-1])
        offsets = dense(keys[:, -8:], name + ".promotion").transpose(0, 2, 1) * divisor
        offsets = offsets[:, :3] + offsets[:, 3:4]
        promotions = np.stack([logits[:, -16:-8, -8:] + offsets[:, p:p+1] for p in range(3)], axis=-1)
        expanded_logits = np.concatenate([logits.reshape(-1, 4096), promotions.reshape(-1, 192)], axis=-1)
        output[name] = expanded_logits[:, policy_map] / divisor
    for name in ("result", "q", "st"):
        embedded = mish(dense(x, name + ".embed")).reshape(len(x), -1)
        hidden = mish(dense(embedded, name + ".hidden"))
        value = dense(hidden, name + ".output")
        if name == "result":
            output["wdl"] = value
        else:
            output[name + ".value"] = np.tanh(value[:, 0])
            output[name + ".error"] = 1 / (1 + np.exp(-dense(hidden, name + ".error")[:, 0]))
            if cfg.value_bins:
                output[name + ".category"] = dense(hidden, name + ".category")
    return output


def flattened(output):
    return {name + "." + key if isinstance(value, dict) else name: item
            for name, value in output.items()
            for key, item in (value.items() if isinstance(value, dict) else [(None, value)])}


def exercise_relative_parameters(model, seed=83):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith((".rq", ".rk", ".rv")):
                parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.15)


def parameter_audit():
    cases = {}
    for name in ("pilot", "cf6m", "cf240m"):
        spec = json.loads(Path(f"configs/{name}.json").read_text())
        cases[name] = ModelConfig(**spec["model"])
    # Source supports independent attention width; this candidate is a hypothesis,
    # never silently installed as the paper's missing large-model configuration.
    cases["cf240m_attention1536_hypothesis"] = replace(cases["cf240m"], attention_dim=1536)
    reports = {}
    for name, cfg in cases.items():
        with torch.device("meta"):
            model = Chessformer(cfg)
        groups = {"input": 0, "attention_projections": 0, "position_tables": 0,
                  "feedforward_and_norms": 0, "policy_heads": 0, "value_heads": 0}
        for parameter_name, parameter in model.named_parameters():
            if parameter_name.startswith("blocks"):
                if ".attn." in parameter_name:
                    group = "attention_projections" if any(s in parameter_name for s in (".qkv.", ".proj.")) else "position_tables"
                else:
                    group = "feedforward_and_norms"
            elif parameter_name.startswith(("policy", "soft_policy")):
                group = "policy_heads"
            elif parameter_name.startswith(("result", "q.", "st.")):
                group = "value_heads"
            else:
                group = "input"
            groups[group] += parameter.numel()
        beta = (8 * cfg.layers) ** -0.25
        reports[name] = {"configuration": asdict(cfg), "parameter_groups": groups,
                         "total_parameters": sum(groups.values()),
                         "deepnorm_beta": beta, "trainer_to_canonical_initial_variance_ratio": 1 / beta}
    return reports


def verify_model():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    maps = source_maps()
    errors = {"float32": {}, "float64": {}}
    cases = 0
    for width in (12, 20):
        cfg = ModelConfig(d_model=12, attention_dim=width, layers=2, heads=4,
                          d_ff=19, value_dim=3, value_hidden=9, value_bins=8,
                          deepnorm="trainer", shaw_cross_term=False)
        model = Chessformer(cfg)
        exercise_relative_parameters(model)
        inputs = torch.randn(2, 64, 112)
        for dtype in (torch.float32, torch.float64):
            model, inputs = model.to(dtype), inputs.to(dtype)
            expected = numpy_forward(model, inputs.numpy(), maps)
            for implementation in ("dense", "factorized"):
                candidate = Chessformer(replace(cfg, attention_impl=implementation)).to(dtype)
                candidate.load_state_dict(model.state_dict())
                with torch.no_grad():
                    actual = flattened(candidate(inputs, return_attention=True))
                for key, value in actual.items():
                    error = float(np.max(np.abs(value.numpy() - expected[key])))
                    name = str(dtype).split(".")[-1]
                    errors[name][key] = max(errors[name].get(key, 0), error)
                    np.testing.assert_allclose(value.numpy(), expected[key], atol=8e-6 if dtype == torch.float32 else 1e-11, rtol=1e-6 if dtype == torch.float32 else 1e-11)
                cases += 1
    sample = torch.empty(768, 1024, dtype=torch.float64)
    beta = (8 * 15) ** -0.25
    keras_variance_scaling_(sample, beta, seed=42)
    variance = 2 * beta / sum(sample.shape)
    files = ["chessformer/model.py", "chessformer/model_reference.py"]
    return {"scope": "source-guided NumPy comparison; TensorFlow runtime and real checkpoints not tested",
            "training_commit": "7b3ee3fcdb49d8abeb034718935b099134c11882",
            "cases": cases, "nonzero_relative_tables": True, "attention_widths": [12, 20],
            "maximum_absolute_errors": errors,
            "initializer": {"sample_shape": list(sample.shape), "requested_variance": variance,
                            "observed_variance": float(sample.var(unbiased=False)),
                            "absolute_max": float(sample.abs().max()),
                            "truncation_bound": 2 * np.sqrt(variance) / 0.87962566103423978,
                            "tensorflow_seed_draws_reproduced": False},
            "parameter_audit": parameter_audit(),
            "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/model-conformance.json")
    args = parser.parse_args()
    report = verify_model()
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "parameter_audit"}, indent=2))
